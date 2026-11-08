"""The probe's choice of windows (D44) — pure, no DB or model."""

from __future__ import annotations

import json

from entity_graph.config import Config
from entity_graph.extract.base import ExtractionResult, RelationProposal
from entity_graph.probe import format_probe, probe

CONFIG = Config(chunk_max_sentences=2, chunk_overlap_sentences=0, chunk_max_chars=10_000)
# Six one-line sentences: windows are sentences 1-2, 3-4 and 5-6.
TEXT = ("Ada Lovelace was born in London. She studied mathematics. "
        "Lovelace later wrote notes. The notes were published. "
        "Lovelace worked with Charles Babbage. Babbage built engines.")


class Reading:
    """Proposes Lovelace WORKS_FOR Babbage wherever both are named; records windows."""

    name = "reading"

    def __init__(self) -> None:
        self.windows: list[str] = []

    def extract(self, text, doc_key=""):
        self.windows.append(text)
        relations = []
        if "Lovelace" in text and "Babbage" in text:
            relations.append(RelationProposal("Lovelace", "Charles Babbage", "WORKS_FOR",
                                              text.strip(), 0.9))
        return ExtractionResult(doc_key, [], relations)


def _gold(tmp_path, evidence: str):
    fact = {"doc": "x", "subject": "Ada Lovelace", "subject_aliases": ["Lovelace"],
            "relation": "WORKS_FOR", "object": "Charles Babbage",
            "object_aliases": ["Babbage"], "modality": "fact", "role": "fact",
            "evidence": evidence}
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(fact) + "\n", encoding="utf-8")
    return path


def test_a_quote_inside_a_window_reads_only_that_window(tmp_path) -> None:
    extractor = Reading()
    result = probe(_gold(tmp_path, "Lovelace worked with Charles Babbage."), TEXT, CONFIG,
                   extractor)
    assert result.windows == 1 and result.name_windows == 0
    assert result.found == [True]


def test_a_quote_wider_than_a_window_falls_back_to_windows_naming_both_ends(tmp_path) -> None:
    extractor = Reading()
    # Sentences 2-5: no two-sentence window holds the whole quote.
    quote = ("She studied mathematics. Lovelace later wrote notes. The notes were published. "
             "Lovelace worked with Charles Babbage.")
    result = probe(_gold(tmp_path, quote), TEXT, CONFIG, extractor)
    assert result.windows == 1 and result.name_windows == 1
    assert [w.strip() for w in extractor.windows] == [
        "Lovelace worked with Charles Babbage. Babbage built engines."]
    assert result.found == [True]
    assert "(1 by name)" in format_probe(result)
