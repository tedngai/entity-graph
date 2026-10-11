"""Entity-type filtering tests — no DB or network.

Excluded ontology types (default DATE) are dropped at the extraction boundary
so they never become graph nodes. See DECISIONS.md D18.
"""

from entity_graph.config import load_config
from entity_graph.extract.base import (
    ExtractionResult,
    Mention,
    RelationProposal,
    is_excluded_type,
)
from entity_graph.pipeline import filter_extraction


def test_is_excluded_type() -> None:
    exclude = frozenset({"DATE"})
    assert is_excluded_type("DATE", exclude)
    assert is_excluded_type("date", exclude)  # case-insensitive
    assert not is_excluded_type("PERSON", exclude)
    assert not is_excluded_type(None, exclude)
    assert not is_excluded_type("", exclude)
    assert not is_excluded_type("DATE", frozenset())


def test_filter_extraction_drops_excluded_mentions_and_proposals() -> None:
    result = ExtractionResult(
        doc_key="d",
        mentions=[
            Mention("1928", "DATE", 0, 4),
            Mention("T. V. Soong", "PERSON", 5, 16),
            Mention("Shanghai", "PLACE", 17, 25),
        ],
        relations=[
            RelationProposal("T. V. Soong", "Shanghai", "LOCATED_IN",
                             source_type="PERSON", target_type="PLACE"),
            RelationProposal("T. V. Soong", "1928", "WORKS_FOR",
                             source_type="PERSON", target_type="DATE"),
            RelationProposal("1928", "Shanghai", "LOCATED_IN",
                             source_type="DATE", target_type="PLACE"),
        ],
    )
    filtered = filter_extraction(result, frozenset({"DATE"}))
    assert [m.surface for m in filtered.mentions] == ["T. V. Soong", "Shanghai"]
    assert [r.target_surface for r in filtered.relations] == ["Shanghai"]


def test_filter_extraction_keeps_untyped_endpoints() -> None:
    # A backend that does not type endpoints cannot be filtered here.
    result = ExtractionResult(
        doc_key="d",
        relations=[RelationProposal("A", "1928", "RELATED_TO")],
    )
    filtered = filter_extraction(result, frozenset({"DATE"}))
    assert len(filtered.relations) == 1


def test_filter_extraction_empty_exclude_is_identity() -> None:
    result = ExtractionResult(
        doc_key="d",
        mentions=[Mention("1928", "DATE", 0, 4)],
    )
    assert filter_extraction(result, frozenset()) is result


def test_config_exclude_entity_types_default_and_override(monkeypatch) -> None:
    monkeypatch.delenv("EXCLUDE_ENTITY_TYPES", raising=False)
    assert load_config().exclude_entity_types == frozenset({"DATE"})

    monkeypatch.setenv("EXCLUDE_ENTITY_TYPES", "date, concept")
    assert load_config().exclude_entity_types == frozenset({"DATE", "CONCEPT"})

    monkeypatch.setenv("EXCLUDE_ENTITY_TYPES", "")
    assert load_config().exclude_entity_types == frozenset()
