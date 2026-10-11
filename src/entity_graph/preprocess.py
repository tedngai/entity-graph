"""OCR text preprocessing, applied at ingest before storage and extraction.

The OCR pipeline that produced our source documents (e.g. the Shanghai
Capitalists book) wraps every page in Markdown scaffolding and leaves inline
footnote markers behind. Extraction and sentence splitting work on prose, so we
strip the scaffolding and rejoin the prose **before** the text is stored. Doing
this at ingest keeps mention offsets and evidence quotes pointing at the same
(cleaned) text, so provenance (invariant #1) still holds; the original file
stays on disk and is recorded in ``documents.source_uri``.

What ``clean_ocr_text`` fixes:

* the OCR preamble (``# OCR Results`` plus its metadata) and the per-page
  markers (``## Page N: imageNNNNN.jpg``), ``---`` rules, and code fences;
* inline footnote markers -- ``\\textsuperscript{41}`` and Unicode superscript
  digits -- and ``\\textit{...}`` wrappers (the text inside is kept);
* page breaks that fall mid-sentence are rejoined, so sentences and entity
  names are not cut in half;
* line/page-break hyphenation (``Shang-\\nhai`` -> ``Shanghai``) using the
  document itself as a dictionary: the hyphen is dropped only when the joined
  form occurs elsewhere in the document, so real hyphenated names
  (``Yueh-sheng``, ``Li-shih``) are preserved;
* typographic quotes/dashes are normalised to ASCII and whitespace is
  collapsed.

The detector is conservative: ``looks_like_ocr`` requires an ``# OCR Results``
heading or a ``## Page N: <file>.<img>`` marker, so ordinary Markdown is left
untouched.
"""

from __future__ import annotations

import re

# --- scaffolding ----------------------------------------------------------
_OCR_HEADING = re.compile(r"^#\s*OCR Results\s*$", re.IGNORECASE | re.MULTILINE)
_PAGE_MARKER = re.compile(
    r"^##\s*Page\s+\d+\s*:\s*\S+\.(?:jpe?g|png|tiff?|gif|bmp)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_PAGE_HEADING = re.compile(r"^##\s*Page\s+\d+\s*:.*$", re.MULTILINE)
_PAGE_BLOCK = re.compile(
    r"^##\s*Page\s+\d+\s*:[^\n]*\n+```[^\n]*\n(.*?)\n```",
    re.MULTILINE | re.DOTALL,
)
_META_LINE = re.compile(
    r"^(?:Source folder|Processed on|Total pages)\s*:.*$", re.MULTILINE
)
_RULE = re.compile(r"^-{3,}\s*$", re.MULTILINE)
_FENCE = re.compile(r"^```.*$", re.MULTILINE)

# --- inline markers -------------------------------------------------------
_TEX_SUPERSCRIPT = re.compile(r"\\textsuperscript\{[^}]*\}")
_TEX_WRAPPER = re.compile(r"\\text[a-zA-Z]*\{([^}]*)\}")
# Unicode superscript digits/letters used as footnote references.
_UNICODE_SUPERSCRIPT = re.compile(r"[\u00b9\u00b2\u00b3\u2070-\u207f]+")

# --- hyphenation ----------------------------------------------------------
_HYPHEN_BREAK = re.compile(r"(?P<head>[A-Za-z]+)-(?P<gap>\s+)(?P<tail>[a-z][A-Za-z]*)")
_WORD = re.compile(r"[A-Za-z]{3,}")

# Sentence-final characters: if a page ends with one of these, the next page
# starts a new paragraph rather than continuing the sentence.
_TERMINAL = tuple(".!?\"')]\u201d\u2019\u2026")


def looks_like_ocr(text: str) -> bool:
    """True when ``text`` carries OCR page scaffolding (or an OCR preamble)."""
    return bool(_OCR_HEADING.search(text)) or bool(_PAGE_MARKER.search(text))


def _strip_inline(text: str) -> str:
    text = _TEX_SUPERSCRIPT.sub("", text)
    text = _TEX_WRAPPER.sub(r"\1", text)  # keep the wrapped words
    text = _UNICODE_SUPERSCRIPT.sub("", text)
    return text


def _strip_scaffolding(text: str) -> str:
    """Fallback for OCR text that has no fenced page blocks."""
    text = _META_LINE.sub("", text)
    text = _OCR_HEADING.sub("", text)
    text = _PAGE_HEADING.sub("", text)
    text = _RULE.sub("", text)
    text = _FENCE.sub("", text)
    return text


def _join_pages(pages: list[str]) -> str:
    """Rejoin pages; a page that ends mid-sentence continues on the next page."""
    out = ""
    for page in pages:
        if not out:
            out = page
            continue
        if out.rstrip().endswith(_TERMINAL):
            out += "\n\n" + page
        else:
            out += " " + page
    return out


def _dehyphenate(text: str) -> str:
    """Join line/page-break hyphenation using the document as its dictionary.

    The hyphen is dropped only when the concatenated form appears elsewhere in
    the document (``Shang-hai`` -> ``Shanghai`` because "Shanghai" is common),
    which preserves genuine hyphenated names (``Yueh-sheng``) that do not.
    """
    words = {match.group(0).casefold() for match in _WORD.finditer(text)}

    def repl(match: re.Match[str]) -> str:
        head, tail = match.group("head"), match.group("tail")
        joined = (head + tail).casefold()
        if len(joined) >= 4 and joined in words:
            return head + tail
        return head + "-" + tail

    return _HYPHEN_BREAK.sub(repl, text)


def _normalize(text: str) -> str:
    text = (
        text.replace("\u2018", "'").replace("\u2019", "'")
        .replace("\u201c", '"').replace("\u201d", '"')
        .replace("\u2013", "-").replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_ocr_text(text: str) -> str:
    """Return cleaned prose for an OCR document (see module docstring)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    blocks = _PAGE_BLOCK.findall(text)
    if blocks:
        pages = [_strip_inline(block.strip()) for block in blocks]
        text = _join_pages(pages)
    else:
        text = _strip_inline(_strip_scaffolding(text))
    text = _dehyphenate(text)
    return _normalize(text)
