"""Load unstructured documents (.txt/.md/.pdf) into the documents table."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .config import Config, load_config
from .db import connect
from .preprocess import clean_ocr_text, looks_like_ocr

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt": "txt", ".md": "md", ".pdf": "pdf"}


def read_document(path: Path) -> tuple[str, str] | None:
    """Return (text, content_type) or None if unsupported/unreadable."""
    suffix = path.suffix.lower()
    content_type = SUPPORTED_SUFFIXES.get(suffix)
    if content_type is None:
        return None
    if content_type == "pdf":
        try:
            from pypdf import PdfReader  # optional extra [ingest-pdf]
        except ImportError:
            log.warning(
                "Skipping %s (PDF support missing: pip install -e .[ingest-pdf])", path
            )
            return None
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text, "pdf"
    return path.read_text(encoding="utf-8", errors="replace"), content_type


def ingest_directory(input_dir: str | Path, config: Config | None = None) -> list[int]:
    """Ingest every supported file under input_dir. Idempotent via sha256 checksum."""
    config = config or load_config()
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    doc_ids: list[int] = []
    with connect(config) as conn, conn.cursor() as cur:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            parsed = read_document(path)
            if parsed is None:
                continue
            text, content_type = parsed
            if not text.strip():
                log.info("Skipping empty file %s", path)
                continue
            if config.preprocess_ocr and looks_like_ocr(text):
                cleaned = clean_ocr_text(text)
                log.info(
                    "Preprocessed OCR text for %s (%d -> %d chars)",
                    path.name,
                    len(text),
                    len(cleaned),
                )
                text = cleaned
            checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cur.execute(
                """
                INSERT INTO documents
                    (doc_key, title, source_uri, content_type, raw_text, checksum)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (checksum) DO NOTHING
                RETURNING id
                """,
                (str(path), path.stem, path.resolve().as_uri(), content_type, text, checksum),
            )
            row = cur.fetchone()
            if row:
                doc_ids.append(row[0])
                log.info("Ingested %s -> document %s", path.name, row[0])
            else:
                log.info("Skipped %s (duplicate content)", path.name)
    return doc_ids
