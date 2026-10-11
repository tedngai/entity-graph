"""spaCy backend tests.

Skipped automatically when spaCy or the ``en_core_web_sm`` model is absent, so
the suite still runs with only the base dependencies (AGENTS.md §5).
"""

import pytest

pytest.importorskip("spacy")

from entity_graph.extract.spacy_extractor import SpacyExtractor  # noqa: E402


def _extractor() -> SpacyExtractor:
    try:
        return SpacyExtractor()
    except RuntimeError:
        pytest.skip("en_core_web_sm is not installed")


def test_spacy_extracts_entities_with_exact_offsets() -> None:
    text = "Maria Chen founded GreenGrid Energy in Berlin."
    result = _extractor().extract(text, "doc")

    types = {m.surface: m.entity_type for m in result.mentions}
    assert types.get("Maria Chen") == "PERSON"
    assert types.get("GreenGrid Energy") == "ORG"
    assert types.get("Berlin") == "PLACE"
    for mention in result.mentions:
        assert text[mention.start:mention.end] == mention.surface


def test_spacy_derives_relation_from_dependency_verb() -> None:
    text = "Maria Chen founded GreenGrid Energy in Berlin."
    result = _extractor().extract(text, "doc")

    founded = [r for r in result.relations if r.relation == "FOUNDED"]
    assert founded, [r.relation for r in result.relations]
    assert any(r.source_surface == "Maria Chen" and r.target_surface == "GreenGrid Energy"
               for r in founded)
    for relation in result.relations:
        assert relation.evidence


def test_spacy_relation_pairs_are_deduplicated() -> None:
    text = "Maria Chen founded GreenGrid Energy. Maria Chen leads GreenGrid Energy."
    result = _extractor().extract(text, "doc")
    keys = [(r.source_surface, r.target_surface, r.relation) for r in result.relations]
    assert len(keys) == len(set(keys))
