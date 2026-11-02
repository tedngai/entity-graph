"""Entity type by majority vote across its mentions (D27).

An entity used to keep the type of whichever mention created it: in the S-1,
``Claude`` (PRODUCT 214, ORG 123, PERSON 109 mentions) stayed PERSON and
``Anthropic`` (ORG 105, PRODUCT 5) stayed PRODUCT. The stored type is now the
plurality of its mentions' types, re-voted as mentions are linked, and
``build`` checks a relation against the voted types of both endpoints as well
as the proposing mention's.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping

from .extract.ontology import RelationSpec, canonical_type, legal_relations

log = logging.getLogger(__name__)

# Mention types that carry no vote while any typed mention exists.
_UNTYPED = {"", "MISC"}


def vote_type(counts: Mapping[str | None, int], current: str | None = None) -> str | None:
    """The plurality type of ``counts``; a tie keeps ``current`` if it is tied.

    MISC and empty types only win when no mention has a real type. Other ties
    break alphabetically so the result does not depend on row order.
    """
    typed = {t: n for t, n in counts.items() if n > 0 and (t or "") not in _UNTYPED}
    pool = typed or {t: n for t, n in counts.items() if n > 0}
    if not pool:
        return current
    top = max(pool.values())
    tied = sorted(t or "" for t, n in pool.items() if n == top)
    if (current or "") in tied:
        return current
    return tied[0] or None


def entity_types_allow(
    relation: str,
    source_type: str | None,
    target_type: str | None,
    specs: tuple[RelationSpec, ...] | None = None,
) -> bool:
    """False only when both entity types are known and the relation is illegal."""
    if not canonical_type(source_type) or not canonical_type(target_type):
        return True
    return relation in legal_relations(source_type, target_type, specs)


def revote_entity(cur, entity_id: int) -> str | None:
    """Re-vote one entity's type from its mentions; returns the new type if changed."""
    cur.execute("SELECT entity_type FROM entities WHERE id = %s", (entity_id,))
    row = cur.fetchone()
    if not row:
        return None
    current = row[0]
    cur.execute(
        "SELECT entity_type, count(*) FROM mentions WHERE entity_id = %s GROUP BY entity_type",
        (entity_id,),
    )
    voted = vote_type(dict(cur.fetchall()), current)
    if voted == current:
        return None
    cur.execute("UPDATE entities SET entity_type = %s WHERE id = %s", (voted, entity_id))
    return voted


def retype_all(cur) -> list[tuple[int, str, str | None, str | None]]:
    """Re-vote every live entity. Returns ``(id, name, old, new)`` per change."""
    cur.execute(
        "SELECT entity_id, entity_type, count(*) FROM mentions "
        "WHERE entity_id IS NOT NULL GROUP BY entity_id, entity_type"
    )
    counts: dict[int, Counter] = {}
    for entity_id, entity_type, n in cur.fetchall():
        counts.setdefault(entity_id, Counter())[entity_type] += n
    cur.execute("SELECT id, canonical_name, entity_type FROM entities WHERE merged_into IS NULL")
    changes = []
    for entity_id, name, current in cur.fetchall():
        if entity_id not in counts:
            continue
        voted = vote_type(counts[entity_id], current)
        if voted != current:
            cur.execute("UPDATE entities SET entity_type = %s WHERE id = %s", (voted, entity_id))
            changes.append((entity_id, name, current, voted))
    return changes


def type_illegal_facts(cur, specs: tuple[RelationSpec, ...] | None = None) -> list[tuple]:
    """Current facts whose relation the endpoints' entity types do not allow.

    Rows are ``(fact_id, source, relation, target, source_type, target_type)``.
    """
    cur.execute(
        """
        SELECT f.id, se.canonical_name, r.relation, te.canonical_name,
               se.entity_type, te.entity_type
        FROM facts f
        JOIN relationships r ON r.id = f.relationship_id
        JOIN entities se ON se.id = r.source_id
        JOIN entities te ON te.id = r.target_id
        WHERE f.is_current
        ORDER BY f.id
        """
    )
    return [
        row for row in cur.fetchall()
        if not entity_types_allow(row[2], row[4], row[5], specs)
    ]
