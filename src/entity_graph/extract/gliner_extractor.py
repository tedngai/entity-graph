"""GLiNER2 extractor backend — the primary extraction path (PROGRESS P4.4).

GLiNER2 (fastino) is a schema-conditioned encoder doing NER, classification,
structured extraction and relation extraction in one forward pass.
Research: docs/RESEARCH.md §3. Requires: ``pip install "gliner2[local]"`` (torch
backend); set ``EXTRACTION_BACKEND=gliner2`` to use.

Verified against gliner2 2.1.0 (2026-10-10):
  * ``AutoExtractor`` is the loader for current checkpoints (both the span and
    boundary architectures); ``GLiNER2.from_pretrained`` is the legacy span path.
  * schema: ``model.create_schema().entities([...]).relations([...])``
  * ``model.extract(text, schema, threshold=..., include_confidence=True,
    include_spans=True)`` returns
        {"entities": {label: [{text, confidence, start, end}, ...]},
         "relation_extraction": {rel: [{"head": {...}, "tail": {...},
                                        "confidence": ...}, ...]}}
Device is chosen from ``GLINER_DEVICE`` (auto | cpu | cuda); on this machine the
RTX 2000 Ada runs the base model in well under a second per short document.
"""

from __future__ import annotations

import logging

from ..chunking import split_sentences
from ..config import load_config
from .base import ExtractionResult, Mention, RelationProposal

log = logging.getLogger(__name__)

DEFAULT_MODEL = "fastino/gliner2.5-base-v1"

# Zero-shot labels: extend freely per domain — no retraining needed.
ENTITY_LABELS = ["person", "organization", "location", "event", "product", "concept", "date"]

_LABEL_TO_TYPE = {
    "person": "PERSON",
    "organization": "ORG",
    "location": "PLACE",
    "event": "EVENT",
    "product": "CONCEPT",
    "concept": "CONCEPT",
    "date": "DATE",
}

RELATION_LABELS = [
    "works for", "located in", "founded", "leads", "partnered with",
    "acquired", "invested in", "member of", "subsidiary of",
]


def _sentence_index(spans: list[tuple[int, int]], position: int) -> int | None:
    for index, (start, end) in enumerate(spans):
        if start <= position < end:
            return index
    return None


def _evidence_for(text: str, head_start: int, tail_start: int) -> str:
    """Evidence quote spanning the head and tail spans.

    The relation is asserted somewhere between the two arguments, so we quote
    from the start of the head's sentence to the end of the tail's sentence
    rather than just the head's line.
    """
    spans = split_sentences(text)
    if not spans:
        return ""
    head_index = _sentence_index(spans, head_start)
    tail_index = _sentence_index(spans, tail_start)
    if head_index is None and tail_index is None:
        return ""
    if head_index is None:
        head_index = tail_index
    if tail_index is None:
        tail_index = head_index
    lo, hi = sorted((head_index, tail_index))
    return text[spans[lo][0]:spans[hi][1]].strip()


class GlinerExtractor:
    name = "gliner2"

    def __init__(
        self,
        model: str | None = None,
        threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from gliner2 import AutoExtractor  # lazy: heavy (torch) dependency
        except ImportError as exc:
            raise RuntimeError(
                'GLiNER2 is not installed. Run: pip install "gliner2[local]" '
                "(then, for CUDA: pip install torch --index-url "
                "https://download.pytorch.org/whl/cu130)"
            ) from exc

        config = load_config()
        self.model_name = model or config.gliner_model or DEFAULT_MODEL
        self.threshold = config.gliner_threshold if threshold is None else threshold
        self.device = device or config.gliner_device or "auto"

        import torch

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {"map_location": self.device} if self.device == "cuda" else {}
        self._model = AutoExtractor.from_pretrained(self.model_name, **kwargs)
        log.info("GLiNER2 %s loaded on %s", self.model_name, self.device)

    def extract(self, text: str, doc_key: str = "") -> ExtractionResult:
        schema = (
            self._model.create_schema()
            .entities(ENTITY_LABELS)
            .relations(RELATION_LABELS)
        )
        result = self._model.extract(
            text,
            schema,
            threshold=self.threshold,
            include_confidence=True,
            include_spans=True,
        )

        mentions: list[Mention] = []
        for label, items in (result.get("entities") or {}).items():
            entity_type = _LABEL_TO_TYPE.get(str(label).lower(), "MISC")
            for item in items or []:
                surface = str(item.get("text", "")).strip()
                if not surface:
                    continue
                mentions.append(
                    Mention(
                        surface=surface,
                        entity_type=entity_type,
                        start=int(item.get("start", 0)),
                        end=int(item.get("end", 0)),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )

        relations: list[RelationProposal] = []
        for relation, items in (result.get("relation_extraction") or {}).items():
            relation_name = str(relation).strip().upper().replace(" ", "_")
            for item in items or []:
                head = item.get("head") or {}
                tail = item.get("tail") or {}
                source = str(head.get("text", "")).strip()
                target = str(tail.get("text", "")).strip()
                if not (source and target and relation_name):
                    continue
                evidence = _evidence_for(
                    text, int(head.get("start", 0)), int(tail.get("start", 0))
                )
                relations.append(
                    RelationProposal(
                        source_surface=source,
                        target_surface=target,
                        relation=relation_name,
                        evidence=evidence,
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )

        log.info(
            "GLiNER2 extracted %d mentions, %d relations for %s",
            len(mentions), len(relations), doc_key or "document",
        )
        return ExtractionResult(doc_key=doc_key, mentions=mentions, relations=relations)
