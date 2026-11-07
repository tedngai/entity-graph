"""LLM relation reader: relations GLiNER found the entities for but not the link (D33).

GLiNER2 tags most entity spans but scores relations pair by pair, so it misses
what needs the passage read as a whole: "Ernst & Young LLP, independent
registered public accounting firm", "legal matters will be passed upon for us
by Fenwick & West LLP", list members with no verb of their own. The reader
hands an LLM one window, the entity strings GLiNER found there, who "we" is
(D28) and the profile's closed relation set, and asks which relations the text
states among those entities. Every answer is checked: both ends must be listed
entities, the relation must be in the profile and legal for their types, and
the evidence must be a verbatim quote (else the sentences spanning both ends
are quoted). The gate judges the result like any other proposal.
"""

from __future__ import annotations

import json
import logging
import re
import threading

from ..config import load_config
from .base import Mention, RelationProposal
from .gliner_extractor import _evidence_for
from .llm import LlmExtractor, parse_json_object
from .ontology import Ontology, RelationSpec, stored_relation

log = logging.getLogger(__name__)

# What each relation means, head -> tail. GLiNER ignores descriptions; an LLM
# needs the direction spelled out.
RELATION_MEANINGS: dict[str, str] = {
    "works_for": "head is or was employed by, or holds or held an officer or staff role at, "
                 "tail; a board seat alone is member_of, not works_for",
    "located_in": "head is located, headquartered or has an office in tail; the state it is "
                  "incorporated in does not count, and an address counts only for its owner",
    "founded": "head founded or co-founded tail",
    "leads": "head is the chief executive, president, chair, manager, managing member or head "
             "of tail (including chair of a board committee); other officers (finance, "
             "operations, science) are works_for, and heading a team inside tail is not "
             "leading tail",
    "partnered_with": "head and tail have a stated partnership or collaboration; a provider or "
                      "customer with no stated partnership is supplies",
    "acquired": "head acquired or agreed to acquire tail",
    "invested_in": "head invested in, or holds equity bought in, tail, including a named "
                   "beneficial owner of tail's stock",
    "member_of": "head is a member, director or trustee of tail (a board, committee, company, "
                 "trust or group)",
    "subsidiary_of": "head is a subsidiary of, or wholly owned by, tail; a trust or fund that "
                     "only holds shares or voting rights is not a subsidiary",
    "supplies": "head sells or provides goods, services, capacity or facilities to tail, "
                "including a customer named as such; lenders and agents of a loan are not "
                "suppliers",
    "sued": "head filed a lawsuit or claim against tail",
    "underwriter_of": "head is an underwriter of tail's securities offering; arrangers and "
                      "bookrunners of a loan are not underwriters",
    "auditor_of": "head is the independent auditor (registered public accounting firm) of tail",
    "advisor_to": "head advises tail, including as its legal counsel; lenders and agents of a "
                  "loan are not advisors",
    "develops": "head develops or makes tail",
    "lends_to": "head lends money to tail, or is a lender, administrative agent, collateral "
                "agent or arranger of a loan, credit facility or notes extended to tail",
}

# A biography often gives the title alone ("has served as a Trustee"); the
# organization is named earlier in the window.
SPLIT_ENDS = """The two ends can be in different sentences: a title with no organization beside
it ("has served as a Trustee") belongs to the organization the text is
discussing, if the text names it."""

SYSTEM_PROMPT = (
    "You read documents and report the relationships they state. "
    "Return ONLY a JSON object, with no prose and no code fences."
)

USER_PROMPT = """\
{self_reference}Entities found in the text (use these exact strings):
{entities}

Relations (head -> tail):
{relations}

List every relation from the list above that the text states as fact between
two of the listed entities. Skip relations that are only planned, possible,
hypothetical, negated or your own inference. The head and tail must fit the
types shown for the relation.

{split_ends}

Return JSON: {{"relations": [{{"head": "<entity>", "relation": "<relation name>",
"tail": "<entity>", "evidence": "<verbatim quote from the text, one or two sentences>"}}]}}
Return {{"relations": []}} if the text states none.

TEXT:
{text}
"""

CONFIDENCE = 0.6
# Running totals over every reader call in this process (calls, tokens).
USAGE = {"calls": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
_USAGE_LOCK = threading.Lock()
_SPACE = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def _locate_quote(text: str, quote: str) -> str | None:
    """The quote as it appears in ``text`` (whitespace-insensitive), else None."""
    if not quote:
        return None
    if quote in text:
        return quote
    words = [re.escape(w) for w in quote.split()]
    if not words:
        return None
    match = re.search(r"\s+".join(words), text)
    return match.group(0) if match else None


# Name parts too common to show that a quote names a particular entity.
_COMMON_PARTS = {
    "inc", "corp", "corporation", "company", "co", "llc", "lp", "llp", "ltd", "plc", "pbc",
    "the", "and", "group", "holdings", "board", "fund", "trust", "partners", "capital",
    "securities", "bank", "national", "association", "management", "international",
}


def _names(quote: str, surface: str) -> bool:
    """Whether ``quote`` names ``surface``: in full, or by a distinctive part of
    the name ("Mr. Hutchins" for Glenn Hutchins, "J.P. Morgan" for J.P. Morgan
    Securities LLC)."""
    if surface.casefold() in quote.casefold():
        return True
    parts = [p for p in re.findall(r"[\w&'’.-]+", surface)
             if p[:1].isupper() and len(p.strip(".")) >= 3
             and p.strip(".").casefold() not in _COMMON_PARTS]
    return any(re.search(rf"(?<![\w.]){re.escape(p)}(?!\w)", quote) for p in parts)


def _count(usage: dict, failed: bool = False) -> None:
    with _USAGE_LOCK:
        USAGE["calls"] += 1
        USAGE["failed"] += int(failed)
        for key in ("prompt_tokens", "completion_tokens"):
            USAGE[key] += int(usage.get(key) or 0)


class RelationReader:
    """Ask an LLM for the profile's relations among a window's entities."""

    name = "llm_reader"

    def __init__(self, ontology: Ontology, llm: LlmExtractor | None = None) -> None:
        self.specs: dict[str, RelationSpec] = {s.name: s for s in ontology.relation_specs}
        self.labels_for_type: dict[str, set[str]] = {}
        for label, entity_type in ontology.label_to_type.items():
            self.labels_for_type.setdefault(entity_type, set()).add(label)
        if llm is None:
            # pratt-medium is a reasoning model: uncapped it reasoned 32,768
            # tokens and answered nothing; at "low" it answers in ~1,400 tokens
            # and still finds counsel (with thinking off it did not).
            config = load_config()
            effort = config.llm_reasoning_effort
            llm = LlmExtractor(timeout=config.llm_timeout,
                               extra={"reasoning_effort": effort} if effort else None)
        self.llm = llm

    @property
    def enabled(self) -> bool:
        return self.llm.enabled

    def _relation_lines(self) -> str:
        return "\n".join(
            f"- {name}: {RELATION_MEANINGS.get(name, name.replace('_', ' '))} "
            f"[head: {'/'.join(spec.head)}; tail: {'/'.join(spec.tail)}]"
            for name, spec in self.specs.items()
        )

    @staticmethod
    def _self_reference(self_name: str | None, self_terms: tuple[str, ...]) -> str:
        if not self_name:
            return ""
        terms = ", ".join(f'"{t}"' for t in self_terms) or '"we", "us", "our"'
        return (f"In this text {terms} refer to {self_name}; report those "
                f'relations with "{self_name}" as the entity.\n\n')

    def _ask(self, prompt: str) -> dict | None:
        """One LLM call, parsed; None (and counted as failed) on any failure."""
        usage: dict = {}
        try:
            content, usage = self.llm._chat_full(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}])
            data = parse_json_object(content)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            _count(usage, failed=True)
            log.warning("%s failed on a window: %s", self.name, exc)
            return None
        _count(usage)
        return data

    def read(self, text: str, mentions: list[Mention], self_name: str | None = None,
             self_terms: tuple[str, ...] = ()) -> list[RelationProposal]:
        """Relations the LLM reads among ``mentions`` in ``text``; [] on any failure."""
        labels: dict[str, set[str]] = {}
        for m in mentions:
            labels.setdefault(m.surface, set()).update(self.labels_for_type.get(m.entity_type, ()))
        if self_name:
            labels.setdefault(self_name, set()).add("organization")
        labels = {s: ls for s, ls in labels.items() if ls}
        if len(labels) < 2:
            return []
        entities = "\n".join(f'- "{s}" ({"/".join(sorted(ls))})' for s, ls in labels.items())
        prompt = USER_PROMPT.format(self_reference=self._self_reference(self_name, self_terms),
                                    entities=entities, relations=self._relation_lines(),
                                    split_ends=SPLIT_ENDS, text=text)
        data = self._ask(prompt)
        return self._check(data.get("relations") or [], labels, text, self_name) if data else []

    def _check(self, items: list, labels: dict[str, set[str]], text: str,
               self_name: str | None = None) -> list[RelationProposal]:
        """The answered relations that pass every check (ends, relation, types, quote).

        A quote must name an end other than the author, who is often only "we"
        (D38): "The underwriters are committed to purchase all the shares
        offered by us" names no underwriter, so the gate rightly finds the
        relation unstated. Such a quote is replaced by the sentences that do
        name the ends."""
        by_fold = {s.casefold(): s for s in labels}
        proposals: list[RelationProposal] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            head = by_fold.get(str(item.get("head", "")).strip().casefold())
            tail = by_fold.get(str(item.get("tail", "")).strip().casefold())
            name = str(item.get("relation", "")).strip().lower().replace(" ", "_")
            spec = self.specs.get(name)
            if not (head and tail and spec) or head == tail:
                continue
            head_labels = labels[head] & set(spec.head)
            tail_labels = labels[tail] & set(spec.tail)
            if not (head_labels and tail_labels):
                continue
            evidence = _locate_quote(text, str(item.get("evidence", "")).strip())
            others = [end for end in (head, tail) if end != self_name]
            if (evidence is None or not _squash(evidence)
                    or not any(_names(evidence, end) for end in others)):
                # An end absent from the window (the author, read from "we")
                # takes the other end's place rather than the window's start.
                h, t = text.find(head), text.find(tail)
                h, t = (h if h >= 0 else t), (t if t >= 0 else h)
                evidence = _evidence_for(text, max(h, 0), max(t, 0), (head, tail))
            proposals.append(RelationProposal(
                source_surface=head, target_surface=tail, relation=stored_relation(name),
                evidence=evidence, confidence=CONFIDENCE,
                source_type=sorted(head_labels)[0], target_type=sorted(tail_labels)[0],
                extractor=self.name,
            ))
        return proposals
