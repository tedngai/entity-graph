"""LLM relation reader — fake LLM, no network. See DECISIONS.md D33."""

from __future__ import annotations

import json

from entity_graph.doc_profiles import get_document_profile
from entity_graph.extract.base import Mention
from entity_graph.extract.relation_reader import RelationReader

TEXT = ("Certain legal matters will be passed upon for us by Fenwick & West LLP. "
        "Ernst & Young LLP, independent registered public accounting firm, audited "
        "our financial statements.")


class FakeLlm:
    enabled = True

    def __init__(self, relations: list[dict]) -> None:
        self.reply = json.dumps({"relations": relations})
        self.last_usage: dict = {}
        self.prompts: list[str] = []

    def _chat_full(self, messages, temperature=0.0):
        self.prompts.append(messages[-1]["content"])
        return self.reply, {"prompt_tokens": 10, "completion_tokens": 5}


def _mention(surface: str, entity_type: str = "ORG") -> Mention:
    start = TEXT.index(surface)
    return Mention(surface, entity_type, start, start + len(surface), 0.9)


MENTIONS = [_mention("Fenwick & West LLP"), _mention("Ernst & Young LLP")]


def _reader(relations: list[dict]) -> tuple[RelationReader, FakeLlm]:
    llm = FakeLlm(relations)
    return RelationReader(get_document_profile("sec_filing").ontology, llm), llm


def test_reader_keeps_checked_relations_and_quotes_the_text() -> None:
    reader, llm = _reader([
        {"head": "fenwick & west llp", "relation": "advisor_to", "tail": "CoreWeave",
         "evidence": "passed upon for us by Fenwick &  West LLP."},
        {"head": "Ernst & Young LLP", "relation": "AUDITOR_OF", "tail": "CoreWeave",
         "evidence": "not in the text"},
    ])
    proposals = reader.read(TEXT, MENTIONS, "CoreWeave", ("we", "us", "our"))
    assert [(p.source_surface, p.relation, p.target_surface) for p in proposals] == [
        ("Fenwick & West LLP", "ADVISOR_TO", "CoreWeave"),
        ("Ernst & Young LLP", "AUDITOR_OF", "CoreWeave"),
    ]
    assert proposals[0].evidence == "passed upon for us by Fenwick & West LLP."
    assert {p.extractor for p in proposals} == {"llm_reader"}  # provenance, not gliner2
    assert proposals[1].evidence in TEXT  # unquotable evidence falls back to the text
    assert '"we", "us", "our" refer to CoreWeave' in llm.prompts[0]


def test_reader_drops_unlisted_entities_unknown_relations_and_illegal_types() -> None:
    reader, _ = _reader([
        {"head": "Deloitte", "relation": "auditor_of", "tail": "CoreWeave", "evidence": ""},
        {"head": "Fenwick & West LLP", "relation": "represents", "tail": "CoreWeave",
         "evidence": ""},
        {"head": "Fenwick & West LLP", "relation": "works_for", "tail": "CoreWeave",
         "evidence": ""},  # works_for needs a person head
    ])
    assert reader.read(TEXT, MENTIONS, "CoreWeave") == []


def test_reader_skips_windows_with_fewer_than_two_entities() -> None:
    reader, llm = _reader([])
    assert reader.read(TEXT, MENTIONS[:1]) == []
    assert llm.prompts == []
