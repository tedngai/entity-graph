"""LLM escalation wrapper (PROGRESS P4.5, decision D2).

The cheap extractor (baseline / spaCy / GLiNER2) runs first. If its result looks
uncertain, the same text is re-extracted by an LLM and the two results are
merged. Escalation is **opt-in** (``EXTRACTION_ESCALATE=true`` or
``--escalate``) and disabled automatically when no LLM key is configured, so the
default pipeline never spends on LLM calls.

An uncertain result is one where the cheap pass:
  * found no mentions at all, or
  * found no relations, or
  * returned mostly low-confidence mentions (below ``escalation_confidence``), or
  * produced very sparse relational structure (fewer than ``min_relation_density``
    relations per mention — e.g. spaCy's small model on a relation-rich text).

Merging keeps every cheap-pass item and adds LLM items that are not already
present (mentions keyed by span, relations by (source, target, relation)), so
provenance is preserved and duplicates are collapsed.
"""

from __future__ import annotations

import logging

from .base import ExtractionResult
from .llm import LlmExtractor

log = logging.getLogger(__name__)


class EscalatingExtractor:
    """Wrap a primary extractor and fall back to an LLM on uncertain documents."""

    def __init__(self, primary, llm: LlmExtractor | None = None,
                 confidence_threshold: float = 0.5,
                 min_relation_density: float = 0.25) -> None:
        self.primary = primary
        self.llm = llm or LlmExtractor()
        self.confidence_threshold = confidence_threshold
        self.min_relation_density = min_relation_density
        self.name = f"{primary.name}+llm"

    def _uncertain(self, result: ExtractionResult) -> bool:
        if not result.mentions:
            return True
        if not result.relations:
            return True
        low = sum(1 for m in result.mentions if m.confidence < self.confidence_threshold)
        if low / len(result.mentions) > 0.5:
            return True
        density = len(result.relations) / len(result.mentions)
        return density < self.min_relation_density

    def extract(self, text: str, doc_key: str = "") -> ExtractionResult:
        result = self.primary.extract(text, doc_key)
        if not self.llm.enabled:
            return result
        if not self._uncertain(result):
            return result

        log.info("Escalating %s to LLM (%s)", doc_key or "document", self.llm.model)
        try:
            extra = self.llm.extract(text, doc_key)
        except Exception as exc:  # noqa: BLE001 - escalation must never break the run
            log.warning("LLM escalation failed for %s: %s", doc_key or "document", exc)
            return result
        return self._merge(result, extra)

    @staticmethod
    def _merge(base: ExtractionResult, extra: ExtractionResult) -> ExtractionResult:
        mentions = {(m.surface, m.start, m.end): m for m in base.mentions}
        for mention in extra.mentions:
            mentions.setdefault((mention.surface, mention.start, mention.end), mention)
        relations = {
            (r.source_surface, r.target_surface, r.relation): r for r in base.relations
        }
        for relation in extra.relations:
            relations.setdefault(
                (relation.source_surface, relation.target_surface, relation.relation),
                relation,
            )
        return ExtractionResult(
            doc_key=base.doc_key,
            mentions=list(mentions.values()),
            relations=list(relations.values()),
        )
