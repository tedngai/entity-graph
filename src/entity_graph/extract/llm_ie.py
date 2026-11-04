"""LLM-only extraction: the LLM finds the entities as well as the relations (D34).

Design C of the extraction trial. The relation reader (D33) only links
entities GLiNER found; this backend asks the same model, with the same
profile relations and self-reference, to find the entities itself. The prompt
asks for specific named entities only: the precision audit (D33) found most
promoted errors had a generic endpoint ("employees", "the company"). The
registrant's own board and committees are the exception, since the gold sets
treat them as nodes.

Every entity must be found verbatim in the window, so each mention carries
offsets (invariant #1); relations pass the reader's checks. Windows run in the
pipeline's worker pool (``concurrent``), and "we" is passed in per window
(``reads_self_reference``) instead of rewriting the text (D28).
"""

from __future__ import annotations

import re

from .base import ExtractionResult, Mention
from .llm import LlmExtractor
from .ontology import Ontology
from .relation_reader import CONFIDENCE, SPLIT_ENDS, RelationReader, _locate_quote

USER_PROMPT = """\
{self_reference}Entity types:
{types}

Relations (head -> tail):
{relations}

Find the entities in the text, then every relation from the list above that
the text states as fact between two of them.

Entities are specific, named things: a person, a company, fund, trust, bank,
agency, court or university, a place, a named product or model. Do not return
generic nouns, roles or groups ("employees", "the company", "our customers",
"underwriters", "chief executive officer", "certain investors"). The one
exception is {self_name}'s own board of directors and its committees, which you
may return as written ("board of directors", "audit committee").
Copy each entity's text exactly as it appears in the text.

Skip relations that are only planned, possible, hypothetical, negated or your
own inference. The head and tail must fit the types shown for the relation.

{split_ends}

Return JSON: {{"entities": [{{"text": "<verbatim>", "type": "<type>"}}],
"relations": [{{"head": "<entity>", "relation": "<relation name>", "tail": "<entity>",
"evidence": "<verbatim quote from the text, one or two sentences>"}}]}}
Return {{"entities": [], "relations": []}} if the text names none.

TEXT:
{text}
"""

_SPACE = re.compile(r"\s+")


class LlmIeExtractor(RelationReader):
    """Entities and relations from one LLM call per window."""

    name = "llm_ie"
    concurrent = True
    reads_self_reference = True

    def __init__(self, ontology: Ontology, llm: LlmExtractor | None = None) -> None:
        super().__init__(ontology, llm)
        self.label_to_type = dict(ontology.label_to_type)
        self.descriptions = dict(ontology.entity_descriptions)

    def _type_lines(self) -> str:
        return "\n".join(f"- {label}: {self.descriptions.get(label, label)}"
                         for label in self.label_to_type)

    def extract(self, text: str, doc_key: str = "", self_name: str | None = None,
                self_terms: tuple[str, ...] = ()) -> ExtractionResult:
        prompt = USER_PROMPT.format(
            self_reference=self._self_reference(self_name, self_terms),
            types=self._type_lines(), relations=self._relation_lines(),
            self_name=self_name or "the author", split_ends=SPLIT_ENDS, text=text)
        data = self._ask(prompt)
        if not data:
            return ExtractionResult(doc_key, [], [])
        mentions: list[Mention] = []
        labels: dict[str, set[str]] = {}
        for item in data.get("entities") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("type", "")).strip().lower()
            entity_type = self.label_to_type.get(label)
            surface = _locate_quote(text, str(item.get("text", "")).strip())
            if not (entity_type and surface and _SPACE.sub("", surface)):
                continue
            start = text.find(surface)
            mentions.append(Mention(surface, entity_type, start, start + len(surface),
                                    CONFIDENCE))
            labels.setdefault(surface, set()).add(label)
        if self_name:
            labels.setdefault(self_name, set()).add("organization")
        # Relations may name an entity with other whitespace than the text;
        # point them at the anchored surface.
        anchored = {_SPACE.sub(" ", s).casefold(): s for s in labels}
        items = []
        for item in data.get("relations") or []:
            if isinstance(item, dict):
                item = dict(item)
                for end in ("head", "tail"):
                    key = _SPACE.sub(" ", str(item.get(end, ""))).strip().casefold()
                    item[end] = anchored.get(key, item.get(end, ""))
                items.append(item)
        relations = self._check(items, labels, text)
        return ExtractionResult(doc_key, mentions, relations)
