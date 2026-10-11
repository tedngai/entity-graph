"""Entity resolution: map mention surface forms to canonical entities.

v1 strategy (cheap, deterministic): exact canonical name -> alias -> pg_trgm
similarity. When a Jev (SystemOne) client is enabled, a *fuzzy* (trigram-only)
candidate is no longer accepted blindly: the pair is submitted to Jev's
entity-alignment Score question and routed per the official cookbook
(PROGRESS P5.3):

    merge    -> link the mention to the candidate and record an alias
    curator  -> create a new entity and queue the pair for human review
    separate -> create a new entity

Exact and alias hits short-circuit (no call). When Jev is disabled or a call
fails, the previous deterministic behaviour is preserved.
"""

from __future__ import annotations

import json
import logging

from .chunking import split_sentences
from .config import Config, load_config
from .db import connect
from .extract.jev import SystemOneClient, align_pair

log = logging.getLogger(__name__)

TRIGRAM_MIN_SIMILARITY = 0.45


def find_exact_or_alias(cur, surface: str) -> int | None:
    """Entity id for an exact canonical-name or alias hit, or None."""
    cur.execute(
        "SELECT id FROM entities WHERE merged_into IS NULL AND canonical_name ILIKE %s "
        "ORDER BY id LIMIT 1",
        (surface,),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "SELECT entity_id FROM entity_aliases WHERE alias ILIKE %s ORDER BY entity_id LIMIT 1",
        (surface,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def find_trigram_candidate(cur, surface: str) -> tuple[int, str, str | None] | None:
    """Best fuzzy (pg_trgm) candidate as (id, canonical_name, entity_type), or None."""
    cur.execute(
        "SELECT id, canonical_name, entity_type FROM entities WHERE merged_into IS NULL "
        "AND similarity(canonical_name, %s) > %s "
        "ORDER BY similarity(canonical_name, %s) DESC LIMIT 1",
        (surface, TRIGRAM_MIN_SIMILARITY, surface),
    )
    return cur.fetchone()


def find_candidate(cur, surface: str) -> int | None:
    """Best existing entity for a surface form (exact/alias/trigram), or None."""
    exact = find_exact_or_alias(cur, surface)
    if exact is not None:
        return exact
    fuzzy = find_trigram_candidate(cur, surface)
    return fuzzy[0] if fuzzy else None


def add_alias(cur, entity_id: int, alias: str, source: str = "extractor") -> None:
    cur.execute(
        "INSERT INTO entity_aliases (entity_id, alias, source) VALUES (%s, %s, %s) "
        "ON CONFLICT (entity_id, alias) DO NOTHING",
        (entity_id, alias, source),
    )


def _set_type_if_missing(cur, entity_id: int, entity_type: str | None) -> None:
    if entity_type:
        cur.execute(
            "UPDATE entities SET entity_type = COALESCE(entity_type, %s) "
            "WHERE id = %s AND entity_type IS NULL",
            (entity_type, entity_id),
        )


def _normalize_name(surface: str) -> str:
    """Collapse internal whitespace (GLiNER2 can span a hard line break)."""
    return " ".join(surface.split())


def _create_entity(cur, surface: str, entity_type: str | None) -> int:
    canonical = _normalize_name(surface) or surface
    cur.execute(
        "INSERT INTO entities (canonical_name, entity_type) VALUES (%s, %s) RETURNING id",
        (canonical, entity_type),
    )
    entity_id = cur.fetchone()[0]
    add_alias(cur, entity_id, canonical, source="extractor")
    if canonical != surface:
        add_alias(cur, entity_id, surface, source="extractor")
    log.debug("Created entity %s for surface %r", entity_id, surface)
    return entity_id


def _sentence_at(text: str, position: int) -> str:
    for start, end in split_sentences(text):
        if start <= position < end:
            return text[start:end].strip()
    return ""


def _mention_context(cur, document_id: int, char_start: int) -> str:
    cur.execute("SELECT raw_text FROM documents WHERE id = %s", (document_id,))
    row = cur.fetchone()
    if not row or not row[0]:
        return ""
    return _sentence_at(row[0], char_start)


def _entity_context(cur, entity_id: int) -> str:
    """A representative sentence for an existing entity (its first mention)."""
    cur.execute(
        "SELECT m.document_id, m.char_start FROM mentions m "
        "WHERE m.entity_id = %s ORDER BY m.id LIMIT 1",
        (entity_id,),
    )
    row = cur.fetchone()
    if not row:
        return ""
    return _mention_context(cur, row[0], row[1])


def _queue_curator(cur, entity_id: int, candidate_id: int, verdict) -> None:
    cur.execute(
        "INSERT INTO curator_queue (entity_id, candidate_id, score, sidecars, reason) "
        "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (entity_id, candidate_id) DO NOTHING",
        (entity_id, candidate_id, verdict.score, json.dumps(verdict.sidecars),
         "Jev alignment score == 1 (uncertain)"),
    )


def ensure_entity(cur, surface: str, entity_type: str | None = None,
                  client: SystemOneClient | None = None,
                  document_id: int | None = None,
                  char_start: int | None = None) -> int:
    """Return the canonical entity id for a surface form, creating one if needed.

    Exact/alias hits return immediately. A trigram-only candidate is accepted
    as-is unless a Jev client is enabled, in which case the pair is routed by
    Jev's alignment verdict (merge / curator / separate).
    """
    exact = find_exact_or_alias(cur, surface)
    if exact is not None:
        _set_type_if_missing(cur, exact, entity_type)
        return exact

    fuzzy = find_trigram_candidate(cur, surface)
    if fuzzy is not None and client is not None and client.enabled:
        candidate_id, candidate_name, candidate_type = fuzzy
        mention_profile: dict = {"name": surface, "kind": entity_type}
        if document_id is not None and char_start is not None:
            context = _mention_context(cur, document_id, char_start)
            if context:
                mention_profile["context"] = context
        candidate_profile: dict = {"name": candidate_name, "kind": candidate_type}
        candidate_context = _entity_context(cur, candidate_id)
        if candidate_context:
            candidate_profile["context"] = candidate_context

        verdict = align_pair(client, mention_profile, candidate_profile)
        if verdict is not None:
            if verdict.outcome == "merge":
                add_alias(cur, candidate_id, surface, source="jev")
                _set_type_if_missing(cur, candidate_id, entity_type)
                log.info("Jev merge %r -> entity %s (score %.2f, %s)",
                         surface, candidate_id, verdict.score, verdict.sidecars)
                return candidate_id
            if verdict.outcome == "curator":
                new_id = _create_entity(cur, surface, entity_type)
                _queue_curator(cur, new_id, candidate_id, verdict)
                log.info("Jev curator: %r vs entity %s (score %.2f) queued",
                         surface, candidate_id, verdict.score)
                return new_id
            log.info("Jev separate %r from entity %s (score %.2f)",
                     surface, candidate_id, verdict.score)
            return _create_entity(cur, surface, entity_type)
        # verdict is None (call failed / disabled mid-run): fall back below.

    if fuzzy is not None:
        _set_type_if_missing(cur, fuzzy[0], entity_type)
        return fuzzy[0]

    return _create_entity(cur, surface, entity_type)


def resolve_unresolved(config: Config | None = None) -> int:
    """Link every unresolved mention to a canonical entity. Returns count."""
    config = config or load_config()
    client = SystemOneClient(config)
    if client.enabled:
        log.info("Jev entity alignment enabled (%s, model %s)", client.url, client.model)
    else:
        log.info("Jev disabled (no API key); using exact/alias/trigram only")

    linked = 0
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT m.id, m.surface_form, m.entity_type, m.document_id, m.char_start "
            "FROM mentions m WHERE m.entity_id IS NULL ORDER BY m.id"
        )
        rows = cur.fetchall()
        for mention_id, surface, entity_type, document_id, char_start in rows:
            entity_id = ensure_entity(
                cur, surface, entity_type,
                client=client, document_id=document_id, char_start=char_start,
            )
            cur.execute(
                "UPDATE mentions SET entity_id = %s WHERE id = %s", (entity_id, mention_id)
            )
            linked += 1
    log.info("Resolved %d mentions", linked)
    return linked


def entity_for_surface_in_doc(cur, surface: str, document_id: int) -> int | None:
    """Entity already linked to this surface in this document, if any."""
    cur.execute(
        "SELECT entity_id FROM mentions "
        "WHERE document_id = %s AND surface_form ILIKE %s AND entity_id IS NOT NULL "
        "ORDER BY id LIMIT 1",
        (document_id, surface),
    )
    row = cur.fetchone()
    return row[0] if row else None


def list_curator_queue(config: Config | None = None, limit: int = 100) -> list[tuple]:
    """Pending uncertain pairs routed by Jev alignment (score == 1)."""
    config = config or load_config()
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT q.id, e.canonical_name, e.entity_type,
                   c.canonical_name, c.entity_type, q.score, q.sidecars
            FROM curator_queue q
            JOIN entities e ON e.id = q.entity_id
            JOIN entities c ON c.id = q.candidate_id
            WHERE q.status = 'pending'
            ORDER BY q.id
            LIMIT %s
            """,
            (limit,),
        )
        return cur.fetchall()
