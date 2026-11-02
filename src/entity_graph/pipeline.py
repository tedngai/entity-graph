"""End-to-end orchestration: ingest -> extract -> resolve -> build."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from .chunking import Chunk, chunk_text
from .config import Config, load_config
from .coordination import expand_coordination
from .db import connect
from .doc_profiles import get_document_profile
from .extract.base import (
    ExtractionResult,
    Extractor,
    Mention,
    RelationProposal,
    get_extractor,
    is_excluded_type,
)
from .graph_store import build_from_proposals, record_mention, record_proposal
from .ingest import ingest_directory
from .resolve import ensure_entity, resolve_unresolved
from .self_reference import SelfReference, find_self_reference, rewrite

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


def _extract_window(extractor: Extractor, text: str, doc_key: str,
                    ref: SelfReference | None) -> ExtractionResult:
    """Extract one window, then copy relations across coordinated lists (D30)."""
    return expand_coordination(_read_window(extractor, text, doc_key, ref), text)


def _read_window(extractor: Extractor, text: str, doc_key: str,
                 ref: SelfReference | None) -> ExtractionResult:
    """Extract one window; with a self-reference, also extract the rewritten
    text, map its mentions and evidence back to the original, and keep both
    readings (D28). Neither wins everywhere: "our Chief Financial Officer"
    needs the name, but "our audit committee" read as "CoreWeave's audit
    committee" pulls the member edge onto the company."""
    plain = extractor.extract(text, doc_key)
    if ref is None:
        return plain
    rw = rewrite(text, ref)
    if not rw.edits:
        return plain
    result = extractor.extract(rw.text, doc_key)
    mentions = list(plain.mentions)
    for mention in result.mentions:
        start, end = rw.span(mention.start, mention.end)
        if 0 <= start < end <= len(text):
            mentions.append(replace(mention, surface=text[start:end], start=start, end=end))

    def original(quote: str) -> str:
        idx = rw.text.find(quote) if quote else -1
        if idx < 0:
            return quote
        start, end = rw.span(idx, idx + len(quote))
        return text[start:end]

    relations = list(plain.relations)
    for proposal in result.relations:
        # The author keeps its name as the surface, so build resolves it to
        # the author; other surfaces and the evidence revert to the original.
        source = proposal.source_surface
        target = proposal.target_surface
        relations.append(replace(
            proposal,
            source_surface=source if source == ref.short else original(source),
            target_surface=target if target == ref.short else original(target),
            evidence=original(proposal.evidence),
        ))
    # Both readings find most spans and pairs; keep one of each.
    unique_mentions = {(m.surface, m.start, m.end, m.entity_type): m for m in mentions}
    unique_relations = {}
    for proposal in relations:
        key = (proposal.source_surface, proposal.target_surface, proposal.relation)
        prev = unique_relations.get(key)
        if prev is None or proposal.confidence > prev.confidence:
            unique_relations[key] = proposal
    return ExtractionResult(doc_key=result.doc_key, mentions=list(unique_mentions.values()),
                            relations=list(unique_relations.values()))


def extract_document(extractor: Extractor, text: str, doc_key: str,
                     config: Config, ref: SelfReference | None = None) -> ExtractionResult:
    """Run the extractor over a document, chunking long texts into windows.

    Short documents (a single window) go straight to the extractor. Longer
    documents are extracted window by window, skipping windows the document
    profile marks as boilerplate (D26), and the results merged: mention
    offsets are rebased to the full document and overlap-induced duplicates are
    collapsed (mentions by span, relations by (source, target, relation)).
    When the document defines who "we" is, each window is extracted with
    those terms replaced by the author's name (D28); pass ``ref`` to use a
    definition found elsewhere in the document.
    """
    ref = ref or find_self_reference(text)
    chunks = chunk_text(
        text,
        max_sentences=config.chunk_max_sentences,
        overlap=config.chunk_overlap_sentences,
        max_chars=config.chunk_max_chars,
    )
    if len(chunks) <= 1:
        return _extract_window(extractor, text, doc_key, ref)

    log.info("Chunked %s into %d sentence windows", doc_key or "document", len(chunks))
    profile = get_document_profile(config.doc_profile)
    mentions: dict[tuple, Mention] = {}
    relations: dict[tuple, RelationProposal] = {}
    skipped = 0
    for chunk in chunks:
        if profile.skips(chunk.text):
            skipped += 1
            continue
        result = _extract_window(extractor, chunk.text, doc_key, ref)
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

    if skipped:
        log.info("Skipped %d boilerplate window(s) (profile %s)", skipped, profile.name)
    return ExtractionResult(
        doc_key=doc_key,
        mentions=list(mentions.values()),
        relations=list(relations.values()),
    )


def filter_extraction(result: ExtractionResult, exclude: frozenset[str] | set[str]) -> ExtractionResult:
    """Drop mentions and relation proposals whose ontology type is excluded.

    Excluded types (default: DATE) are not graph nodes. Their surface still
    appears inside fact evidence quotes, so no provenance is lost. Proposals
    are dropped only when an endpoint type is *known* to be excluded; a
    proposal with an untyped endpoint is kept (a backend that does not type
    endpoints cannot be filtered here).
    """
    if not exclude:
        return result
    mentions = [
        m for m in result.mentions if not is_excluded_type(m.entity_type, exclude)
    ]
    relations = [
        r for r in result.relations
        if not is_excluded_type(r.source_type, exclude)
        and not is_excluded_type(r.target_type, exclude)
    ]
    return ExtractionResult(doc_key=result.doc_key, mentions=mentions, relations=relations)


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
            result = filter_extraction(result, config.exclude_entity_types)
            for mention in result.mentions:
                record_mention(cur, doc_id, mention.surface, mention.start, mention.end,
                               mention.entity_type, extractor.name, mention.confidence)
            for proposal in result.relations:
                record_proposal(cur, doc_id, proposal.source_surface, proposal.target_surface,
                                proposal.relation, proposal.evidence, extractor.name,
                                proposal.confidence, proposal.source_type, proposal.target_type)
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
