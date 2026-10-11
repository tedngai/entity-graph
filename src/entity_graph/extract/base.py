"""Extraction contract shared by every backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

ENTITY_TYPES = {"PERSON", "ORG", "PLACE", "CONCEPT", "EVENT", "DATE", "MISC"}


@dataclass
class Mention:
    """A surface form found in a document, with char offsets into the raw text."""

    surface: str
    entity_type: str = "MISC"
    start: int = 0
    end: int = 0
    confidence: float = 1.0


@dataclass
class RelationProposal:
    """A candidate (subject, relation, object) with the evidence sentence."""

    source_surface: str
    target_surface: str
    relation: str
    evidence: str = ""
    confidence: float = 1.0


@dataclass
class ExtractionResult:
    doc_key: str
    mentions: list[Mention] = field(default_factory=list)
    relations: list[RelationProposal] = field(default_factory=list)


@runtime_checkable
class Extractor(Protocol):
    name: str

    def extract(self, text: str, doc_key: str = "") -> ExtractionResult: ...


def get_extractor(backend: str, config=None) -> Extractor:
    """Factory. Heavy dependencies are imported inside each backend module.

    When ``config.escalate`` is set (and the backend is not itself ``llm``), the
    chosen backend is wrapped in an ``EscalatingExtractor`` that falls back to an
    LLM on uncertain documents (PROGRESS P4.5).
    """
    if backend == "baseline":
        from .baseline_extractor import BaselineExtractor

        extractor: Extractor = BaselineExtractor()
    elif backend == "spacy":
        from .spacy_extractor import SpacyExtractor

        extractor = SpacyExtractor()
    elif backend == "gliner2":
        from .gliner_extractor import GlinerExtractor

        extractor = GlinerExtractor()
    elif backend == "llm":
        from .llm import LlmExtractor

        extractor = LlmExtractor()
    else:
        raise ValueError(
            f"Unknown extraction backend {backend!r} "
            "(expected baseline | spacy | gliner2 | llm)"
        )

    if getattr(config, "escalate", False) and backend != "llm":
        from .escalation import EscalatingExtractor

        extractor = EscalatingExtractor(
            extractor, confidence_threshold=config.escalation_confidence
        )
    return extractor
