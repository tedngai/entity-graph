"""Preprocessing tests: pure-python, no DB, no network, no heavy dependencies."""

from entity_graph.preprocess import clean_ocr_text, looks_like_ocr

OCR_SAMPLE = (
    "# OCR Results\n"
    "\n"
    "Source folder: E:\\AI\\OCR\\Some Book\n"
    "\n"
    "Processed on: 2025-03-23 22:33:42\n"
    "\n"
    "Total pages: 2\n"
    "\n"
    "---\n"
    "\n"
    "## Page 1: image00001.jpg\n"
    "\n"
    "```\n"
    "Chiang Kai-shek\u2074 met the Shanghai bankers. Tu Yueh-sheng was a leading figure.\n"
    "\n"
    "The average Shang-\n"
    "```\n"
    "\n"
    "---\n"
    "\n"
    "## Page 2: image00002.jpg\n"
    "\n"
    "```\n"
    "hai quotation of 5 major banks rose. Soong\u2019s policy \\textit{San Min Chu I} mattered.\n"
    "```\n"
)


def test_looks_like_ocr_detects_scaffolding() -> None:
    assert looks_like_ocr(OCR_SAMPLE) is True


def test_looks_like_ocr_ignores_plain_markdown() -> None:
    assert looks_like_ocr("# Notes\n\nMaria Chen founded GreenGrid Energy.") is False


def test_clean_strips_scaffolding() -> None:
    cleaned = clean_ocr_text(OCR_SAMPLE)
    assert "OCR Results" not in cleaned
    assert "Source folder" not in cleaned
    assert "Total pages" not in cleaned
    assert "## Page" not in cleaned
    assert "image00001.jpg" not in cleaned
    assert "```" not in cleaned
    assert "\n---" not in cleaned


def test_clean_removes_footnote_markers() -> None:
    cleaned = clean_ocr_text(OCR_SAMPLE)
    assert "\u2074" not in cleaned  # unicode superscript footnote
    assert "Kai-shek met" in cleaned
    assert "\\textsuperscript" not in cleaned


def test_clean_unwraps_latex_and_keeps_inner_text() -> None:
    cleaned = clean_ocr_text(OCR_SAMPLE)
    assert "\\textit" not in cleaned
    assert "San Min Chu I" in cleaned


def test_clean_normalises_typographic_punctuation() -> None:
    cleaned = clean_ocr_text(OCR_SAMPLE)
    assert "Soong's policy" in cleaned


def test_clean_rejoins_page_break_mid_sentence() -> None:
    # Page 1 ends mid-sentence ("Shang-") and page 2 continues ("hai").
    cleaned = clean_ocr_text(OCR_SAMPLE)
    assert "Shanghai quotation" in cleaned
    assert "Shang- hai" not in cleaned


def test_clean_preserves_real_hyphenated_names() -> None:
    # "Yueh-sheng" occurs elsewhere, so the line-break hyphen is kept.
    cleaned = clean_ocr_text(OCR_SAMPLE)
    assert "Yueh-sheng" in cleaned
    assert "Yuehsheng" not in cleaned


def test_clean_is_idempotent() -> None:
    once = clean_ocr_text(OCR_SAMPLE)
    assert clean_ocr_text(once) == once


def test_clean_keeps_paragraph_breaks_within_a_page() -> None:
    cleaned = clean_ocr_text(OCR_SAMPLE)
    # The blank line inside page 1 stays a paragraph break.
    assert "\n\n" in cleaned
