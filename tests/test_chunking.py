"""Chunking tests: pure-python, no DB, no network, no heavy dependencies."""

from entity_graph.chunking import chunk_text, split_sentences
from entity_graph.config import Config
from entity_graph.extract.baseline_extractor import BaselineExtractor
from entity_graph.pipeline import extract_document


def test_split_sentences_tiles_the_text() -> None:
    text = "One two. Three four! Five six?"
    spans = split_sentences(text)
    assert [text[s:e].strip() for s, e in spans] == [
        "One two.",
        "Three four!",
        "Five six?",
    ]
    assert "".join(text[s:e] for s, e in spans) == text


def test_split_sentences_keeps_abbreviations_together() -> None:
    text = "Dr. Smith met J. Doe in the U.S. today."
    spans = split_sentences(text)
    assert len(spans) == 1
    assert text[spans[0][0]:spans[0][1]].strip() == text


def test_split_sentences_empty() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_split_sentences_tiles_with_blank_lines() -> None:
    text = "First line.\n\nSecond line here.\n\n\nThird one.\n"
    spans = split_sentences(text)
    assert "".join(text[s:e] for s, e in spans) == text
    assert all(text[s:e].strip() for s, e in spans)
    assert len(spans) == 3


def test_split_sentences_keeps_hard_wrapped_prose_together() -> None:
    # A lone newline is intra-sentence whitespace, not a boundary.
    text = "A sentence that wraps\nacross two lines. Another one\nhere too."
    spans = split_sentences(text)
    assert "".join(text[s:e] for s, e in spans) == text
    assert [text[s:e].strip() for s, e in spans] == [
        "A sentence that wraps\nacross two lines.",
        "Another one\nhere too.",
    ]


def test_split_sentences_paragraph_break_is_a_boundary() -> None:
    text = "Title line\n\nBody sentence one. Body sentence two."
    spans = split_sentences(text)
    assert "".join(text[s:e] for s, e in spans) == text
    assert [text[s:e].strip() for s, e in spans] == [
        "Title line",
        "Body sentence one.",
        "Body sentence two.",
    ]


def test_evidence_for_spans_head_and_tail() -> None:
    from entity_graph.extract.gliner_extractor import _evidence_for

    text = "GreenGrid Energy announced a deal. The partner is Nordwind Capital."
    evidence = _evidence_for(text, text.index("GreenGrid"), text.index("Nordwind"))
    assert evidence == text  # both sentences are quoted

    same = "Nordwind Capital, the firm led by Tobias Renner."
    assert _evidence_for(same, same.index("Nordwind"), same.index("Tobias")) == same


def test_chunk_offsets_are_absolute_and_exact() -> None:
    text = " ".join(f"Sentence {i} ends." for i in range(10))
    chunks = chunk_text(text, max_sentences=3, overlap=1, max_chars=10_000)
    assert len(chunks) >= 4
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for chunk in chunks:
        assert chunk.text == text[chunk.start:chunk.end]
        assert chunk.start < chunk.end
        assert len(split_sentences(chunk.text)) <= 3
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)


def test_windows_overlap() -> None:
    text = " ".join(f"Sentence {i} ends." for i in range(10))
    chunks = chunk_text(text, max_sentences=2, overlap=1, max_chars=10_000)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start < prev.end  # shared sentence
        assert text[prev.start:prev.end] == prev.text


def test_single_oversized_sentence_is_emitted_whole() -> None:
    text = "A" + " b" * 3000 + "."
    chunks = chunk_text(text, max_sentences=4, overlap=1, max_chars=50)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert (chunks[0].start, chunks[0].end) == (0, len(text))


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []
    assert chunk_text("  \n ") == []


def test_extract_document_rebases_offsets() -> None:
    text = " ".join(f"Alpha{i} Beta{i} met Gamma{i}." for i in range(8))
    config = Config(chunk_max_sentences=2, chunk_overlap_sentences=1, chunk_max_chars=10_000)
    result = extract_document(BaselineExtractor(), text, "doc", config)

    assert result.mentions
    for mention in result.mentions:
        assert 0 <= mention.start < mention.end <= len(text)
        assert mention.surface == text[mention.start:mention.end]


def test_extract_document_dedupes_overlap() -> None:
    text = " ".join(f"Alpha{i} Beta{i} met Gamma{i}." for i in range(8))
    config = Config(chunk_max_sentences=2, chunk_overlap_sentences=1, chunk_max_chars=10_000)
    result = extract_document(BaselineExtractor(), text, "doc", config)

    mention_keys = [(m.surface, m.start, m.end) for m in result.mentions]
    assert len(mention_keys) == len(set(mention_keys))
    relation_keys = [(r.source_surface, r.target_surface, r.relation) for r in result.relations]
    assert len(relation_keys) == len(set(relation_keys))


def test_extract_document_short_text_matches_plain_extractor() -> None:
    text = "Maria Chen founded GreenGrid Energy in Berlin."
    result = extract_document(BaselineExtractor(), text, "doc", Config())
    plain = BaselineExtractor().extract(text, "doc")
    assert [(m.surface, m.start, m.end) for m in result.mentions] == [
        (m.surface, m.start, m.end) for m in plain.mentions
    ]
