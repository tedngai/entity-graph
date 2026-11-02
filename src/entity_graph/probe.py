"""Fast extraction-recall probe: run the extractor on gold windows only.

A full run (extract -> resolve -> build) takes hours on a filing, and most
missed gold facts die at extraction (no proposal at all). The probe chunks a
document exactly as ``extract`` does, keeps the windows that contain a gold
evidence quote, runs the configured extractor on them, and scores the
proposals against the gold facts. No database, no gate calls.

It measures proposal recall only: precision over a handful of windows says
nothing about the whole document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_text
from .config import Config
from .eval_facts import GoldFact, load_gold, matches

_SPACE = re.compile(r"\s+")


def _flat(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


@dataclass
class ProbeResult:
    facts: list[GoldFact]
    found: list[bool]
    traps: list[GoldFact]
    leaked: list[bool]
    windows: int
    # Proposals matching a gold fact or trap: (proposal, gold item), for --gate.
    matched: list = field(default_factory=list)

    @property
    def recall(self) -> float:
        return sum(self.found) / len(self.found) if self.found else 0.0


def probe(gold_path: str | Path, text: str, config: Config, extractor) -> ProbeResult:
    """Extract the windows holding gold evidence and score proposals."""
    from .pipeline import extract_document
    from .self_reference import find_self_reference

    gold = load_gold(gold_path)
    chunks = chunk_text(
        text,
        max_sentences=config.chunk_max_sentences,
        overlap=config.chunk_overlap_sentences,
        max_chars=config.chunk_max_chars,
    )
    quotes = {_flat(g.evidence) for g in gold if g.evidence}
    wanted = [c for c in chunks if any(q in _flat(c.text) for q in quotes)]
    # Each window goes through extract_document, so any document-level step
    # it applies to a single window applies here too.
    ref = find_self_reference(text)
    proposals = []
    for chunk in wanted:
        proposals.extend(extract_document(extractor, chunk.text, "", config, ref=ref).relations)
    predictions = [
        {"subject": r.source_surface, "relation": r.relation, "object": r.target_surface}
        for r in proposals
    ]
    facts = [g for g in gold if g.role == "fact"]
    traps = [g for g in gold if g.role == "trap"]
    matched = [
        (proposal, g) for proposal, p in zip(proposals, predictions)
        for g in facts + traps if matches(p, g)
    ]
    return ProbeResult(
        matched=matched,
        facts=facts,
        found=[any(matches(p, g) for p in predictions) for g in facts],
        traps=traps,
        leaked=[any(matches(p, g) for p in predictions) for g in traps],
        windows=len(wanted),
    )


def format_probe(result: ProbeResult, *, verbose: bool = False) -> str:
    lines = [
        f"Probe: {result.windows} gold window(s); proposal recall "
        f"{sum(result.found)}/{len(result.facts)} ({result.recall:.0%}); "
        f"traps proposed {sum(result.leaked)}/{len(result.traps)}"
    ]
    if verbose:
        for fact, hit in zip(result.facts, result.found):
            lines.append(f"  {'Y' if hit else '.'} {fact.subject} {fact.relation} {fact.object}")
        for trap, hit in zip(result.traps, result.leaked):
            lines.append(f"  {'L' if hit else '.'} TRAP {trap.subject} {trap.relation} {trap.object}")
    return "\n".join(lines)


def gate_probe(result: ProbeResult, config: Config) -> str:
    """Send the gold-matching proposals to the relation gate; report survivors.

    Facts kept / facts proposed shows gate recall on true candidates; traps
    kept shows what still leaks. Uses the configured Jev client and profile.
    """
    from .extract.gate import should_promote
    from .extract.jev import RELATION_VALIDATION_THRESHOLD, SystemOneClient, judge_relations
    from .extract.profiles import get_profile

    client = SystemOneClient(config)
    if not client.enabled:
        return "Gate probe skipped: no SystemOne key configured."
    claims = []
    for index, (proposal, _gold) in enumerate(result.matched):
        claim = {
            "id": index,
            "subject": proposal.source_surface,
            "relation": proposal.relation,
            "object": proposal.target_surface,
            "evidence": proposal.evidence or "",
            "source_type": proposal.source_type or "",
            "target_type": proposal.target_type or "",
        }
        claims.append(claim)
    verdicts = judge_relations(client, claims, profile=get_profile(config.systemone_profile))
    kept_facts: set[int] = set()
    kept_traps: set[int] = set()
    unanswered = 0
    for index, (_proposal, gold) in enumerate(result.matched):
        judgment = verdicts.get(index)
        if judgment is None:
            unanswered += 1
            continue
        if should_promote(judgment, threshold=RELATION_VALIDATION_THRESHOLD):
            (kept_traps if gold.role == "trap" else kept_facts).add(id(gold))
    proposed_facts = {id(g) for _p, g in result.matched if g.role == "fact"}
    proposed_traps = {id(g) for _p, g in result.matched if g.role == "trap"}
    return (f"Gate: facts kept {len(kept_facts)}/{len(proposed_facts)} proposed "
            f"(of {len(result.facts)}); traps kept {len(kept_traps)}/{len(proposed_traps)}; "
            f"{len(claims)} claims, {unanswered} unanswered")
