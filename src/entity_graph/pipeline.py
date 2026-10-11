"""End-to-end orchestration: ingest -> extract -> resolve -> build."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from .chunking import Chunk, chunk_text
from .config import Config, load_config
from .db import connect
from .extract.base import ExtractionResult, Extractor, Mention, RelationProposal, get_extractor
from .graph_store import build_from_proposals, record_mention, record_proposal
from .ingest import ingest_directory
from .resolve import ensure_entity, resolve_unresolved

log = logging.getLogger(__name__)


def _rebase_mention(mention: Mention, chunk: Chunk) -> Mention | None:
    """Translate a chunk-relative mention into document coordinates.

    Returns None when the mention cannot be anchored (bad offsets and surface
    not found in the chunk), so no unverifiable provenance is ever stored.
    """
    s, e = mention.start, mention.end
    if 0 <= s < e <= len(chunk.text) and chunk.text[s:e] == mention.surface:
        return replace(mention, start=chunk.start + s, end=chunk.start + e)
    if mention.surface:
        idx = chunk.text.find(mention.surface)
        if idx >= 0:
            return replace(
                mention,
                start=chunk.start + idx,
                end=chunk.start + idx + len(mention.surface),
            )
    return None


def extract_document(extractor: Extractor, text: str, doc_key: str,
                     config: Config) -> ExtractionResult:
    """Run the extractor over a document, chunking long texts into windows.

    Short documents (a single window) go straight to the extractor. Longer
    documents are extracted window by window and the results merged: mention
    offsets are rebased to the full document and overlap-induced duplicates are
    collapsed (mentions by span, relations by (source, target, relation)).
    """
    chunks = chunk_text(
        text,
        max_sentences=config.chunk_max_sentences,
        overlap=config.chunk_overlap_sentences,
        max_chars=config.chunk_max_chars,
    )
    if len(chunks) <= 1:
        return extractor.extract(text, doc_key)

    log.info("Chunked %s into %d sentence windows", doc_key or "document", len(chunks))
    mentions: dict[tuple, Mention] = {}
    relations: dict[tuple, RelationProposal] = {}
    for chunk in chunks:
        result = extractor.extract(chunk.text, doc_key)
        for mention in result.mentions:
            rebased = _rebase_mention(mention, chunk)
            if rebased is None:
                log.debug("Dropping unanchorable mention %r in %s", mention.surface, doc_key)
                continue
            key = (rebased.surface, rebased.start, rebased.end, rebased.entity_type)
            prev = mentions.get(key)
            if prev is None or rebased.confidence > prev.confidence:
                mentions[key] = rebased
        for proposal in result.relations:
            key = (proposal.source_surface, proposal.target_surface, proposal.relation)
            prev = relations.get(key)
            if prev is None or proposal.confidence > prev.confidence:
                relations[key] = proposal

    return ExtractionResult(
        doc_key=doc_key,
        mentions=list(mentions.values()),
        relations=list(relations.values()),
    )


def extract_documents(config: Config | None = None, force: bool = False) -> int:
    """Run the configured extractor over documents lacking mentions."""
    config = config or load_config()
    extractor = get_extractor(config.extraction_backend, config)
    processed = 0
    with connect(config) as conn, conn.cursor() as cur:
        if force:
            cur.execute("SELECT id, doc_key, raw_text FROM documents ORDER BY id")
        else:
            cur.execute(
                "SELECT d.id, d.doc_key, d.raw_text FROM documents d "
                "WHERE NOT EXISTS (SELECT 1 FROM mentions m WHERE m.document_id = d.id) "
                "ORDER BY id"
            )
        for doc_id, doc_key, text in cur.fetchall():
            if not text:
                continue
            log.info("Extracting %s with backend %r", doc_key, extractor.name)
            result = extract_document(extractor, text, doc_key, config)
            for mention in result.mentions:
                record_mention(cur, doc_id, mention.surface, mention.start, mention.end,
                               mention.entity_type, extractor.name, mention.confidence)
            for proposal in result.relations:
                record_proposal(cur, doc_id, proposal.source_surface, proposal.target_surface,
                                proposal.relation, proposal.evidence, extractor.name,
                                proposal.confidence)
            processed += 1
    return processed


def build(config: Config | None = None, validate: bool = True) -> int:
    """Promote staged proposals into relationships + facts."""
    return build_from_proposals(config, validate=validate)


def run_pipeline(input_dir: str | Path, config: Config | None = None,
                 validate: bool = True) -> dict:
    """Full pipeline: ingest, extract, resolve mentions, build relationships."""
    config = config or load_config()
    docs = ingest_directory(input_dir, config)
    extracted = extract_documents(config)
    resolved = resolve_unresolved(config)
    edges = build(config, validate=validate)

    summary = {
        "documents_ingested": len(docs),
        "documents_extracted": extracted,
        "mentions_resolved": resolved,
        "relationships_built": edges,
    }
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM entities")
        summary["entities_total"] = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM facts WHERE is_current")
        summary["facts_total"] = cur.fetchone()[0]
    return summary
