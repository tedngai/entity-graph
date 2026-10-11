"""LLM extractor + escalation tests — no network (the chat call is stubbed)."""

import json

import pytest

from entity_graph.config import Config
from entity_graph.extract.base import ExtractionResult, Mention, RelationProposal, get_extractor
from entity_graph.extract.escalation import EscalatingExtractor
from entity_graph.extract.llm import LlmExtractor, parse_json_object


def _llm(monkeypatch, reply: str) -> LlmExtractor:
    extractor = LlmExtractor(model="test/model", api_key="test-key")
    monkeypatch.setattr(extractor, "_chat", lambda messages, temperature=0.0: reply)
    return extractor


def test_parse_json_object_tolerates_fences() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here: {"a": 2} done') == {"a": 2}
    with pytest.raises(ValueError):
        parse_json_object("no json here")


def test_llm_extract_anchors_offsets_and_types(monkeypatch) -> None:
    text = "Maria Chen founded GreenGrid Energy in Berlin."
    reply = json.dumps(
        {
            "entities": [
                {"text": "Maria Chen", "type": "PERSON"},
                {"text": "GreenGrid Energy", "type": "ORG"},
                {"text": "Berlin", "type": "PLACE"},
                {"text": "Nowhere", "type": "PLACE"},  # absent -> dropped
            ],
            "relations": [
                {"source": "Maria Chen", "target": "GreenGrid Energy",
                 "relation": "founded", "evidence": text},
            ],
        }
    )
    result = _llm(monkeypatch, reply).extract(text, "doc")

    assert {m.surface for m in result.mentions} == {"Maria Chen", "GreenGrid Energy", "Berlin"}
    for mention in result.mentions:
        assert text[mention.start:mention.end] == mention.surface
    assert result.relations[0].relation == "FOUNDED"
    assert result.relations[0].evidence == text


def test_llm_disabled_without_key() -> None:
    assert LlmExtractor(model="m", api_key="").enabled is False


def test_escalation_triggers_on_empty_primary(monkeypatch) -> None:
    class EmptyPrimary:
        name = "baseline"

        def extract(self, text, doc_key=""):
            return ExtractionResult(doc_key=doc_key)

    llm = _llm(monkeypatch, json.dumps(
        {"entities": [{"text": "Acme", "type": "ORG"}], "relations": []}
    ))
    wrapper = EscalatingExtractor(EmptyPrimary(), llm=llm)
    assert wrapper.name == "baseline+llm"

    result = wrapper.extract("Acme is here.", "doc")
    assert [m.surface for m in result.mentions] == ["Acme"]


def test_escalation_skips_confident_primary(monkeypatch) -> None:
    class Confident:
        name = "baseline"

        def extract(self, text, doc_key=""):
            return ExtractionResult(
                doc_key=doc_key,
                mentions=[Mention("Acme", "ORG", 0, 4, 0.9)],
                relations=[RelationProposal("Acme", "Beta", "RELATED_TO", "e", 0.9)],
            )

    llm = _llm(monkeypatch, "{}")
    calls = {"n": 0}

    def counting_chat(messages, temperature=0.0):
        calls["n"] += 1
        return "{}"

    monkeypatch.setattr(llm, "_chat", counting_chat)
    wrapper = EscalatingExtractor(Confident(), llm=llm)

    result = wrapper.extract("Acme Beta", "doc")
    assert calls["n"] == 0
    assert len(result.mentions) == 1


def test_escalation_triggers_on_sparse_relations(monkeypatch) -> None:
    class Sparse:
        name = "spacy"

        def extract(self, text, doc_key=""):
            return ExtractionResult(
                doc_key=doc_key,
                mentions=[Mention(f"E{i}", "ORG", 0, 1, 0.9) for i in range(10)],
                relations=[RelationProposal("E0", "E1", "RELATED_TO", "e", 0.9)],
            )

    reply = json.dumps(
        {"entities": [], "relations": [
            {"source": "A", "target": "B", "relation": "X", "evidence": "e"}]}
    )
    wrapper = EscalatingExtractor(Sparse(), llm=_llm(monkeypatch, reply))
    result = wrapper.extract("text", "doc")
    assert any(r.relation == "X" for r in result.relations)


def test_escalation_disabled_without_key() -> None:
    class EmptyPrimary:
        name = "baseline"

        def extract(self, text, doc_key=""):
            return ExtractionResult(doc_key=doc_key)

    wrapper = EscalatingExtractor(EmptyPrimary(), llm=LlmExtractor(model="m", api_key=""))
    assert wrapper.extract("text", "doc").mentions == []


def test_factory_supports_llm_and_escalation() -> None:
    assert get_extractor("llm").name == "llm"
    wrapped = get_extractor("baseline", Config(escalate=True))
    assert wrapped.name == "baseline+llm"
    assert get_extractor("baseline", Config(escalate=False)).name == "baseline"
