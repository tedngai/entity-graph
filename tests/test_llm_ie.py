"""LLM-only extraction backend — fake LLM, no network. See DECISIONS.md D34."""

from __future__ import annotations

import json

from entity_graph.config import Config
from entity_graph.doc_profiles import get_document_profile
from entity_graph.extract.llm_ie import LlmIeExtractor
from entity_graph.pipeline import extract_document

TEXT = ("Certain legal matters will be passed upon for us by Fenwick & West LLP. "
        "Karen Boone serves on the board of directors of Sonos, Inc.")


class FakeLlm:
    enabled = True

    def __init__(self, reply: dict) -> None:
        self.reply = json.dumps(reply)
        self.prompts: list[str] = []

    def _chat_full(self, messages, temperature=0.0):
        self.prompts.append(messages[-1]["content"])
        return self.reply, {"prompt_tokens": 10, "completion_tokens": 5}


def _extractor(reply: dict) -> tuple[LlmIeExtractor, FakeLlm]:
    llm = FakeLlm(reply)
    return LlmIeExtractor(get_document_profile("sec_filing").ontology, llm), llm


def test_llm_ie_anchors_entities_and_checks_relations() -> None:
    extractor, llm = _extractor({
        "entities": [
            {"text": "Fenwick &  West LLP", "type": "organization"},  # whitespace differs
            {"text": "Karen Boone", "type": "person"},
            {"text": "Sonos, Inc.", "type": "organization"},
            {"text": "Acme Corp", "type": "organization"},  # not in the text
            {"text": "legal matters", "type": "widget"},  # unknown type
        ],
        "relations": [
            {"head": "Fenwick & West LLP", "relation": "advisor_to", "tail": "CoreWeave",
             "evidence": "passed upon for us by Fenwick & West LLP."},
            {"head": "Karen Boone", "relation": "member_of", "tail": "Sonos, Inc.",
             "evidence": "Karen Boone serves on the board of directors of Sonos, Inc."},
            {"head": "Acme Corp", "relation": "advisor_to", "tail": "CoreWeave", "evidence": ""},
        ],
    })
    result = extractor.extract(TEXT, "doc", "CoreWeave", ("we", "us", "our"))

    assert [(m.surface, m.entity_type) for m in result.mentions] == [
        ("Fenwick & West LLP", "ORG"), ("Karen Boone", "PERSON"), ("Sonos, Inc.", "ORG")]
    for m in result.mentions:
        assert TEXT[m.start:m.end] == m.surface
    assert [(r.source_surface, r.relation, r.target_surface) for r in result.relations] == [
        ("Fenwick & West LLP", "ADVISOR_TO", "CoreWeave"),
        ("Karen Boone", "MEMBER_OF", "Sonos, Inc."),
    ]
    assert {r.extractor for r in result.relations} == {"llm_ie"}
    assert "generic nouns" in llm.prompts[0] and "CoreWeave's own board" in llm.prompts[0]


UNDERWRITING = ("We are offering the shares through a number of underwriters. "
                "J.P. Morgan Securities LLC and Jefferies LLC are acting as representatives "
                "of the underwriters. The underwriters are committed to purchase all the "
                "common shares offered by us. Glenn Hutchins joined our board in 2024. "
                "Mr. Hutchins is the chair of our audit committee.")


def test_a_quote_must_name_an_end_other_than_the_author() -> None:
    extractor, _ = _extractor({
        "entities": [
            {"text": "J.P. Morgan Securities LLC", "type": "organization"},
            {"text": "Jefferies LLC", "type": "organization"},
            {"text": "Glenn Hutchins", "type": "person"},
            {"text": "audit committee", "type": "organization"},
        ],
        "relations": [
            # Names only "us": replaced by the sentence that names the firm.
            {"head": "J.P. Morgan Securities LLC", "relation": "underwriter_of", "tail": "Acme",
             "evidence": "The underwriters are committed to purchase all the common shares "
                         "offered by us."},
            # Names the firm: kept as given.
            {"head": "Jefferies LLC", "relation": "underwriter_of", "tail": "Acme",
             "evidence": "J.P. Morgan Securities LLC and Jefferies LLC are acting as "
                         "representatives of the underwriters."},
            # "Mr. Hutchins" names Glenn Hutchins by a distinctive part: kept.
            {"head": "Glenn Hutchins", "relation": "leads", "tail": "audit committee",
             "evidence": "Mr. Hutchins is the chair of our audit committee."},
        ],
    })
    result = extractor.extract(UNDERWRITING, "doc", "Acme", ("we", "us", "our"))
    quotes = {r.source_surface: r.evidence for r in result.relations}
    assert quotes["J.P. Morgan Securities LLC"] == (
        "J.P. Morgan Securities LLC and Jefferies LLC are acting as representatives "
        "of the underwriters.")
    assert quotes["Jefferies LLC"].startswith("J.P. Morgan Securities LLC and Jefferies LLC")
    assert quotes["Glenn Hutchins"] == "Mr. Hutchins is the chair of our audit committee."


def test_llm_ie_failure_yields_an_empty_window() -> None:
    class Broken(FakeLlm):
        def _chat_full(self, messages, temperature=0.0):
            raise RuntimeError("LLM returned no content")

    extractor = LlmIeExtractor(get_document_profile("sec_filing").ontology, Broken({}))
    result = extractor.extract(TEXT, "doc")
    assert result.mentions == [] and result.relations == []


def test_pipeline_runs_llm_ie_windows_in_the_pool_without_the_reader(monkeypatch) -> None:
    from entity_graph import pipeline

    monkeypatch.setattr(pipeline, "_relation_reader",
                        lambda config: (_ for _ in ()).throw(AssertionError("reader called")))
    extractor, llm = _extractor({"entities": [{"text": "Karen Boone", "type": "person"}],
                                 "relations": []})
    text = " ".join([TEXT] * 6)
    config = Config(chunk_max_sentences=2, chunk_overlap_sentences=0, chunk_max_chars=10_000)
    result = extract_document(extractor, text, "doc", config)
    assert len(llm.prompts) >= 2  # every window, in the pool
    assert len(result.mentions) == len(llm.prompts)  # one per window, offsets rebased
    assert {text[m.start:m.end] for m in result.mentions} == {"Karen Boone"}
