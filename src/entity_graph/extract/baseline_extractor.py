"""Zero-dependency heuristic extractor.

Purpose: pipeline bring-up, demos, and tests without model downloads.
Method: capitalized token runs -> mentions (type guessed from surface);
same-sentence mention pairs -> RELATED_TO proposals.

Known limits (why this is not the production extractor): no disambiguation,
sentence-initial capitalized words can be missed or over-caught, months/weekday
names surface as mentions. Replace with GLiNER2 (`EXTRACTION_BACKEND=gliner2`)
for real work.
"""

from __future__ import annotations

import re

from .base import ExtractionResult, Mention, RelationProposal

# Two or more capitalized tokens, optionally joined by lowercase connectors.
_MULTI = re.compile(
    r"\b[A-Z][A-Za-z0-9&\-]*(?:\s+(?:(?:of|the|for|and|de|von|van|der)\s+)?"
    r"[A-Z][A-Za-z0-9&\-]*)+"
)
# All-caps acronyms (NATO, UNICEF, ...)
_ACRONYM = re.compile(r"\b[A-Z]{2,8}\b")
# Single capitalized word NOT at a sentence start (preceded by lowercase/comma).
_SINGLE = re.compile(r"(?<=[a-z,] )\b[A-Z][a-z]{2,}\b")

_SENTENCES = re.compile(r"[^.!?\n]+[.!?]?")

_ORG_HINTS = {
    "inc", "corp", "llc", "ltd", "gmbh", "group", "university", "institute",
    "ministry", "agency", "bank", "capital", "partners", "energy", "foundation",
    "association", "council", "company", "commission", "laboratory", "labs",
}


def _infer_type(surface: str) -> str:
    if _ACRONYM.fullmatch(surface):
        return "ORG"
    tokens = surface.split()
    if tokens and tokens[-1].lower().rstrip(".,") in _ORG_HINTS:
        return "ORG"
    if len(tokens) >= 2:
        return "PERSON"
    return "MISC"


def _find_mentions(text: str) -> list[Mention]:
    found: list[tuple[int, int, str]] = []

    def add(match: re.Match) -> None:
        span = match.span()
        if not any(span[0] < e and span[1] > s for s, e, _ in found):
            found.append((span[0], span[1], match.group(0)))

    for pattern in (_MULTI, _ACRONYM, _SINGLE):
        for match in pattern.finditer(text):
            add(match)

    return [
        Mention(surface=surface, entity_type=_infer_type(surface), start=s, end=e, confidence=0.5)
        for s, e, surface in sorted(found)
    ]


def _find_relations(text: str, mentions: list[Mention]) -> list[RelationProposal]:
    """Mentions sharing a sentence are proposed as RELATED_TO (baseline heuristic)."""
    proposals: list[RelationProposal] = []
    seen: set[tuple[str, str, str]] = set()
    for sentence in _SENTENCES.finditer(text):
        lo, hi = sentence.span()
        here = [m for m in mentions if m.start >= lo and m.end <= hi]
        for i, a in enumerate(here):
            for b in here[i + 1:]:
                if a.surface == b.surface:
                    continue
                key = (a.surface, b.surface, "RELATED_TO")
                if key in seen:
                    continue
                seen.add(key)
                proposals.append(
                    RelationProposal(
                        source_surface=a.surface,
                        target_surface=b.surface,
                        relation="RELATED_TO",
                        evidence=sentence.group(0).strip(),
                        confidence=0.4,
                    )
                )
    return proposals


class BaselineExtractor:
    name = "baseline"

    def extract(self, text: str, doc_key: str = "") -> ExtractionResult:
        mentions = _find_mentions(text)
        return ExtractionResult(
            doc_key=doc_key,
            mentions=mentions,
            relations=_find_relations(text, mentions),
        )
