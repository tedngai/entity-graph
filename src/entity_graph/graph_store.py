"""Write path (entities / relationships / facts) and NetworkX read path.

Reads must go through v_graph_nodes / v_graph_edges only (merged entities are
excluded there). See docs/ARCHITECTURE.md contracts.
"""

from __future__ import annotations

import logging
from datetime import date

from .config import Config, load_config
from .db import connect
from .extract.decide import judge_relations as decide_judge_relations
from .extract.gate import should_promote
from .extract.jev import (
    RELATION_VALIDATION_THRESHOLD,
    SystemOneClient,
    judge_relations,
)
from .extract.profiles import get_profile
from .resolve import add_alias, entity_for_surface_in_doc, ensure_entity

log = logging.getLogger(__name__)


def _judge_rows(config: Config, client: SystemOneClient | None, rows: list) -> dict:
    """Jev when a key is set; otherwise the local Decide gate. Empty if neither runs."""
    if client is None or not rows:
        return {}
    claims = [
        {
            "id": pid,
            "subject": src,
            "relation": relation,
            "object": dst,
            "evidence": evidence or "",
            "source_type": source_type or "",
            "target_type": target_type or "",
        }
        for pid, _doc, src, dst, relation, evidence, _ex, _conf, source_type, target_type in rows
    ]
    if client.enabled:
        verdicts = judge_relations(
            client, claims, profile=get_profile(config.systemone_profile)
        )
        log.info("Jev judged %d/%d relation proposals", len(verdicts), len(rows))
        return verdicts
    verdicts = decide_judge_relations(
        claims, model=config.decide_model, device=config.decide_device,
    )
    if verdicts:
        log.info("Decide judged %d/%d relation proposals", len(verdicts), len(rows))
    return verdicts


# ---------------------------------------------------------------------------
# Write path
# ---------------------------------------------------------------------------

def record_mention(cur, document_id: int, surface_form: str, char_start: int,
                   char_end: int, entity_type: str, extractor: str,
                   confidence: float) -> int:
    cur.execute(
        """
        INSERT INTO mentions
            (document_id, surface_form, char_start, char_end, entity_type, extractor, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (document_id, surface_form, char_start, char_end, entity_type, extractor, confidence),
    )
    return cur.fetchone()[0]


def record_proposal(cur, document_id: int, source_surface: str, target_surface: str,
                    relation: str, evidence: str, extractor: str, confidence: float,
                    source_type: str = "", target_type: str = "") -> int:
    """Stage a candidate relation; `build` promotes it to relationship + fact."""
    cur.execute(
        """
        INSERT INTO relation_proposals
            (document_id, source_surface, target_surface, relation, evidence_text,
             extractor, confidence, source_type, target_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (document_id, source_surface, target_surface, relation, evidence, extractor,
         confidence, source_type or None, target_type or None),
    )
    return cur.fetchone()[0]


def upsert_entity(cur, canonical_name: str, entity_type: str | None = None) -> int:
    return ensure_entity(cur, canonical_name, entity_type)


def upsert_relationship(cur, source_id: int, target_id: int, relation: str,
                        is_directed: bool = True) -> int:
    cur.execute(
        """
        INSERT INTO relationships (source_id, target_id, relation, is_directed)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (source_id, target_id, relation) DO UPDATE
            SET relation = EXCLUDED.relation
        RETURNING id
        """,
        (source_id, target_id, relation, is_directed),
    )
    return cur.fetchone()[0]


def record_fact(cur, relationship_id: int, document_id: int, statement: str,
                evidence_text: str, extractor: str, confidence: float,
                valid_from: date | None = None, valid_to: date | None = None) -> int:
    cur.execute(
        """
        INSERT INTO facts
            (relationship_id, document_id, statement, evidence_text, extractor,
             confidence, valid_from, valid_to)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (relationship_id, document_id, statement, evidence_text, extractor,
         confidence, valid_from, valid_to),
    )
    return cur.fetchone()[0]


def build_from_proposals(config: Config | None = None, validate: bool = True) -> int:
    """Promote pending relation_proposals into relationships + facts.

    Surfaces are resolved against mentions of the same document first, then
    through ensure_entity (exact/alias/trigram). When ``validate`` is set, Jev
    (if a key is configured) or GLiNER2.5-Decide (local, no key) judges each
    pending proposal. A refuted claim is marked ``rejected`` instead of being
    promoted. The Jev assertion cutoff is the frozen constant; modality is an
    argmax. Returns the number of edges built.
    """
    config = config or load_config()
    client = SystemOneClient(config) if validate else None
    built = 0
    with connect(config) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, document_id, source_surface, target_surface, relation,
                   evidence_text, extractor, confidence, source_type, target_type
            FROM relation_proposals WHERE status = 'pending' ORDER BY id
            """
        )
        rows = cur.fetchall()
        verdicts = _judge_rows(config, client, rows)

        for (pid, doc_id, src, dst, relation, evidence, extractor, confidence,
             _source_type, _target_type) in rows:
            judgment = verdicts.get(pid)
            assertion = judgment.assertion if judgment is not None else None
            modality = judgment.modality if judgment is not None else None
            gate = judgment.source if judgment is not None else None
            if judgment is not None and not should_promote(
                judgment, threshold=RELATION_VALIDATION_THRESHOLD
            ):
                cur.execute(
                    "UPDATE relation_proposals SET status = 'rejected', validation = %s, "
                    "modality = %s, gate = %s WHERE id = %s",
                    (assertion, modality, gate, pid),
                )
                log.info(
                    "%s rejected proposal %s (%s %s %s, assertion=%s, modality=%s)",
                    gate or "gate", pid, src, relation, dst, assertion, modality,
                )
                continue

            src_id = entity_for_surface_in_doc(cur, src, doc_id) or ensure_entity(cur, src)
            dst_id = entity_for_surface_in_doc(cur, dst, doc_id) or ensure_entity(cur, dst)
            if src_id == dst_id:
                log.debug("Skipping self-relation proposal %s (%s)", pid, src)
            else:
                relationship_id = upsert_relationship(cur, src_id, dst_id, relation)
                statement = f"{src} {relation.lower().replace('_', ' ')} {dst}"
                record_fact(cur, relationship_id, doc_id, statement, evidence or "",
                            extractor, confidence)
                built += 1
            cur.execute(
                "UPDATE relation_proposals SET status = 'promoted', validation = %s, "
                "modality = %s, gate = %s WHERE id = %s",
                (assertion, modality, gate, pid),
            )
    log.info("Built %d relationships from proposals", built)
    return built


# ---------------------------------------------------------------------------
# Read path: Postgres -> NetworkX
# ---------------------------------------------------------------------------

def load_graph(config: Config | None = None):
    """Load the current graph as nx.MultiDiGraph from the graph views."""
    import networkx as nx
    import pandas as pd

    config = config or load_config()
    with connect(config) as conn:
        nodes = pd.read_sql("SELECT * FROM v_graph_nodes", conn)
        edges = pd.read_sql("SELECT * FROM v_graph_edges", conn)

    graph = nx.MultiDiGraph()
    for row in nodes.itertuples(index=False):
        graph.add_node(
            row.id, name=row.name, type=row.type or "MISC",
            description=row.description or "", attributes=row.attributes,
        )
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source_id, row.target_id, key=row.relation,
            relation=row.relation, fact_count=row.fact_count,
            attributes=row.attributes,
        )
    log.info("Loaded graph: %d nodes, %d edges", graph.number_of_nodes(),
             graph.number_of_edges())
    return graph
