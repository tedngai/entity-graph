"""Extraction backends: baseline (zero-dep), spaCy, GLiNER2 (primary)."""

from .base import (
    ExtractionResult,
    Extractor,
    Mention,
    RelationProposal,
    get_extractor,
)

__all__ = [
    "ExtractionResult",
    "Extractor",
    "Mention",
    "RelationProposal",
    "get_extractor",
]
