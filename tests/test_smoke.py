"""Smoke tests: no database, no network, no heavy dependencies required."""

from entity_graph.extract.base import get_extractor
from entity_graph.extract.baseline_extractor import BaselineExtractor

SAMPLE = (
    "Maria Chen founded GreenGrid Energy in Berlin. "
    "The company partnered with Nordwind Capital last year."
)


def test_baseline_finds_multiword_mentions() -> None:
    result = BaselineExtractor().extract(SAMPLE, doc_key="sample")
    surfaces = {m.surface for m in result.mentions}
    assert "Maria Chen" in surfaces
    assert "GreenGrid Energy" in surfaces
    assert all(m.start < m.end for m in result.mentions)
    assert all(m.surface == SAMPLE[m.start:m.end] for m in result.mentions)


def test_baseline_proposes_same_sentence_relations() -> None:
    result = BaselineExtractor().extract(SAMPLE, doc_key="sample")
    pair = {(r.source_surface, r.target_surface) for r in result.relations}
    assert ("Maria Chen", "GreenGrid Energy") in pair
    assert all(r.relation == "RELATED_TO" for r in result.relations)
    assert all(r.evidence for r in result.relations)


def test_unknown_backend_raises() -> None:
    try:
        get_extractor("does-not-exist")
    except ValueError as exc:
        assert "does-not-exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for unknown backend")


def test_jev_disabled_without_key(monkeypatch) -> None:
    import entity_graph.extract.jev as jev

    monkeypatch.delenv("SYSTEMONE_API_KEY", raising=False)
    monkeypatch.delenv("PRATT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = jev.SystemOneClient()
    assert client.enabled is False
    assert jev.align_pair(client, {"name": "A"}, {"name": "B"}) is None
    assert jev.validates_relation(client, "text", "A", "WORKS_FOR", "B") is None
