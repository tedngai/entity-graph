"""Sentence-window chunking for long documents (PROGRESS P3.3).

Long documents are split into overlapping windows of sentences *before*
extraction. Every :class:`Chunk` carries char offsets into the original
document text, so after extraction the backend's chunk-relative mention
offsets can be rebased back to document coordinates and provenance stays
valid (invariant #1 in AGENTS.md).

Windows overlap by a configurable number of sentences so an entity mention or
a relation whose evidence straddles a boundary is still seen by at least one
window. Downstream code dedupes the overlap (see ``pipeline.extract_document``).

This module is pure stdlib and has no heavy dependencies, so it is importable
from tests and from ``entity-graph --help`` without the extraction extras.
"""

from __future__ import annotations

from dataclasses import dataclass

# Sentence terminators, closing punctuation that belongs to the sentence, and
# whitespace that is absorbed into the preceding sentence span.
_TERMINATORS = frozenset(".!?")
_CLOSERS = frozenset("\"')]}»\u201d\u2019")
_WHITESPACE = frozenset(" \t\r\n")

# Tokens whose trailing period is not a sentence boundary. Lower-cased; matched
# against the token immediately before the terminator. Single uppercase letters
# (initials such as "J.") are handled separately in _is_abbreviation.
_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
        "e.g", "i.e", "no", "inc", "ltd", "co", "corp", "dept", "est",
        "u.s", "u.k", "a.m", "p.m", "fig", "al", "approx", "min", "max",
    }
)


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of a document, with offsets into the full text."""

    text: str
    start: int
    end: int
    index: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start


def _is_abbreviation(text: str, dot: int) -> bool:
    """True when the '.' at ``dot`` closes an abbreviation or an initial."""
    j = dot - 1
    while j >= 0 and text[j].isalpha():
        j -= 1
    token = text[j + 1:dot]
    if not token:
        return False
    if len(token) == 1 and token.isupper():
        return True  # initial, e.g. "J. Smith"
    return token.lower() in _ABBREVIATIONS


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` char spans that tile ``text``.

    Boundaries fall after runs of ``.!?`` (plus trailing closers/whitespace) and
    at *paragraph breaks* — runs of whitespace containing at least two newlines.
    A single newline is treated as ordinary intra-sentence whitespace, so
    hard-wrapped prose (lines wrapped at ~80 cols) is not split mid-sentence.
    Whitespace-only runs are folded into the neighbouring span so the spans stay
    contiguous: ``"".join(text[s:e] for s, e in spans) == text`` for any
    non-blank input, and ``text[s:e].strip()`` is never empty.
    """
    raw: list[tuple[int, int]] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _TERMINATORS:
            j = i + 1
            while j < n and text[j] in _TERMINATORS:
                j += 1
            while j < n and text[j] in _CLOSERS:
                j += 1
            # Only a boundary when followed by whitespace or end of text.
            if j < n and text[j] not in _WHITESPACE:
                i = j
                continue
            if ch == "." and _is_abbreviation(text, i):
                i = j
                continue
            raw.append((start, j))
            start = j
            i = j
            continue
        if ch == "\n":
            j = i
            while j < n and text[j] in _WHITESPACE:
                j += 1
            # A boundary only at a paragraph break (a blank line). A lone
            # newline is intra-sentence whitespace for hard-wrapped prose.
            if text[i:j].count("\n") >= 2:
                raw.append((start, j))
                start = j
            i = j
            continue
        i += 1
    if start < n:
        raw.append((start, n))

    # Fold whitespace-only spans into a neighbour so the spans stay contiguous
    # and no empty sentence is ever returned.
    merged: list[list[int]] = []
    pending_start: int | None = None
    for s, e in raw:
        if pending_start is not None:
            s, pending_start = pending_start, None
        if text[s:e].strip():
            merged.append([s, e])
        elif merged:
            merged[-1][1] = e
        else:
            pending_start = s
    if pending_start is not None and merged:
        merged[0][0] = pending_start
    return [(s, e) for s, e in merged]


def chunk_text(
    text: str,
    *,
    max_sentences: int = 8,
    overlap: int = 2,
    max_chars: int = 1200,
) -> list[Chunk]:
    """Split ``text`` into overlapping sentence windows.

    A window grows sentence by sentence until it reaches ``max_sentences`` or
    ``max_chars`` (whichever comes first), always including at least one
    sentence even if that sentence alone exceeds ``max_chars``. The next window
    starts ``overlap`` sentences before the current window ends, so consecutive
    windows share ``overlap`` sentences. Offsets are absolute into ``text``, so
    ``chunk.text == text[chunk.start:chunk.end]`` for every chunk.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    max_sentences = max(1, max_sentences)
    overlap = max(0, min(overlap, max_sentences - 1))

    chunks: list[Chunk] = []
    total = len(sentences)
    i = 0
    while i < total:
        start = sentences[i][0]
        end = sentences[i][1]
        count = 1
        j = i + 1
        while j < total:
            if count >= max_sentences:
                break
            if sentences[j][1] - start > max_chars:
                break
            end = sentences[j][1]
            count += 1
            j += 1
        chunks.append(Chunk(text=text[start:end], start=start, end=end, index=len(chunks)))
        if j >= total:
            break
        step = max(1, count - overlap)
        i += step
    return chunks
