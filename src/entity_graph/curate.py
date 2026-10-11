"""Curation: merge duplicate entities and hide non-actors — never delete.

Both operations respect invariant #2 (never hard-delete entities or facts) and
are reversible:

* ``merge(winner, loser)`` repoints the loser's relationships to the winner (so
  no edge is lost), collapses any relationship that becomes a duplicate, moves
  aliases and mentions over, then sets ``merged_into``. The loser row stays.
* ``hide(entity)`` sets ``is_hidden``; the row and its facts remain, but the
  graph views stop exposing the entity and its edges.

A curation file is JSON::

    {
      "renames": [["Soong", "T. V. Soong"]],
      "merges": [["T. V. Soong", "T. V. SOONG"], ["Tu Yueh-sheng", "Tu"]],
      "hide":   ["central bank", "finance", "banking"],
      "unhide": []
    }

Every name is a canonical name or alias. In a merge the first element is the
winner (the surviving entity). See DECISIONS.md D19.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, load_config
from .db import connect

log = logging.getLogger(__name__)


@dataclass
class CurationSpec:
    """A reversible curation plan: renames, merges, hides, and un-hides."""

    renames: list[tuple[str, str]] = field(default_factory=list)
    merges: list[tuple[str, str]] = field(default_factory=list)
    hide: list[str] = field(default_factory=list)
    unhide: list[str] = field(default_factory=list)


def load_curation(path: str | Path) -> CurationSpec:
    """Parse and validate a curation JSON file (pure; no database)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    renames: list[tuple[str, str]] = []
    for pair in data.get("renames", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"each rename must be [old, new]; got {pair!r}")
        renames.append((str(pair[0]), str(pair[1])))
    merges: list[tuple[str, str]] = []
    for pair in data.get("merges", []):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"each merge must be [winner, loser]; got {pair!r}")
        merges.append((str(pair[0]), str(pair[1])))
    return CurationSpec(
        renames=renames,
        merges=merges,
        hide=[str(x) for x in data.get("hide", [])],
        unhide=[str(x) for x in data.get("unhide", [])],
    )


def resolve_entity(cur, name: str) -> int | None:
    """Live entity id for a canonical name or alias (case-insensitive)."""
    cur.execute(
        """
        SELECT e.id
        FROM entities e
        WHERE e.merged_into IS NULL
          AND (lower(e.canonical_name) = lower(%s)
               OR EXISTS (SELECT 1 FROM entity_aliases a
                          WHERE a.entity_id = e.id AND lower(a.alias) = lower(%s)))
        ORDER BY (lower(e.canonical_name) = lower(%s)) DESC, e.id
        LIMIT 1
        """,
        (name, name, name),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _merge(cur, winner_id: int, loser_id: int) -> dict:
    """Repoint a loser's edges/aliases/mentions to the winner, then mark it."""
    if winner_id == loser_id:
        return {"merged": False, "reason": "same entity"}
    cur.execute("SELECT merged_into, canonical_name FROM entities WHERE id = %s", (loser_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"entity {loser_id} not found")
    if row[0] is not None:
        return {"merged": False, "reason": "already merged"}
    loser_name = row[1]

    # 1. aliases: move the loser's aliases and its canonical name to the winner.
    cur.execute(
        "INSERT INTO entity_aliases (entity_id, alias, source) "
        "SELECT %s, alias, source FROM entity_aliases WHERE entity_id = %s "
        "ON CONFLICT DO NOTHING",
        (winner_id, loser_id),
    )
    cur.execute(
        "INSERT INTO entity_aliases (entity_id, alias, source) VALUES (%s, %s, 'merge') "
        "ON CONFLICT DO NOTHING",
        (winner_id, loser_name),
    )

    # 2. relationships: repoint loser -> winner, collapsing duplicates.
    cur.execute(
        "SELECT id, source_id, target_id, relation FROM relationships "
        "WHERE source_id = %s OR target_id = %s",
        (loser_id, loser_id),
    )
    moved = collapsed = loops = 0
    for rid, src, tgt, rel in cur.fetchall():
        new_src = winner_id if src == loser_id else src
        new_tgt = winner_id if tgt == loser_id else tgt
        if new_src == new_tgt:  # merge turned the edge into a self-loop
            cur.execute("SELECT count(*) FROM facts WHERE relationship_id = %s", (rid,))
            if cur.fetchone()[0]:
                cur.execute(
                    "UPDATE relationships SET source_id = %s, target_id = %s WHERE id = %s",
                    (new_src, new_tgt, rid),
                )
            else:
                cur.execute("DELETE FROM relationships WHERE id = %s", (rid,))
            loops += 1
            continue
        cur.execute(
            "SELECT id FROM relationships WHERE source_id = %s AND target_id = %s "
            "AND relation = %s AND id <> %s ORDER BY id LIMIT 1",
            (new_src, new_tgt, rel, rid),
        )
        existing = cur.fetchone()
        if existing:
            cur.execute(
                "UPDATE facts SET relationship_id = %s WHERE relationship_id = %s",
                (existing[0], rid),
            )
            cur.execute("DELETE FROM relationships WHERE id = %s", (rid,))
            collapsed += 1
        else:
            cur.execute(
                "UPDATE relationships SET source_id = %s, target_id = %s WHERE id = %s",
                (new_src, new_tgt, rid),
            )
            moved += 1

    # 3. provenance: mentions follow the surviving entity.
    cur.execute("UPDATE mentions SET entity_id = %s WHERE entity_id = %s", (winner_id, loser_id))
    # 4. the curator queue no longer has a question about the loser.
    cur.execute(
        "DELETE FROM curator_queue WHERE entity_id = %s OR candidate_id = %s",
        (loser_id, loser_id),
    )
    # 5. mark the loser (row kept; excluded from the views).
    cur.execute("UPDATE entities SET merged_into = %s WHERE id = %s", (winner_id, loser_id))
    log.info("Merged entity %s (%r) into %s", loser_id, loser_name, winner_id)
    return {"merged": True, "moved": moved, "collapsed": collapsed, "self_loops": loops}


def merge_entities(winner_id: int, loser_id: int, config: Config | None = None) -> dict:
    """Merge two entities in their own transaction."""
    with connect(config) as conn, conn.cursor() as cur:
        return _merge(cur, winner_id, loser_id)


def _set_hidden(cur, entity_id: int, hidden: bool) -> bool:
    cur.execute(
        "UPDATE entities SET is_hidden = %s WHERE id = %s AND merged_into IS NULL",
        (hidden, entity_id),
    )
    return cur.rowcount > 0


def _rename(cur, entity_id: int, new_name: str) -> str | None:
    """Rename an entity's canonical name; the old name is kept as an alias."""
    cur.execute("SELECT canonical_name FROM entities WHERE id = %s", (entity_id,))
    old_name = cur.fetchone()[0]
    if old_name == new_name:
        return None
    cur.execute("UPDATE entities SET canonical_name = %s WHERE id = %s", (new_name, entity_id))
    cur.execute(
        "INSERT INTO entity_aliases (entity_id, alias, source) VALUES (%s, %s, 'rename') "
        "ON CONFLICT DO NOTHING",
        (entity_id, old_name),
    )
    return old_name


def apply_curation(
    spec: CurationSpec, config: Config | None = None, dry_run: bool = False
) -> dict:
    """Apply a curation plan. Returns a summary; makes no changes when dry_run."""
    config = config or load_config()
    summary: dict = {
        "renamed": [], "merged": [], "hidden": [], "unhidden": [], "missing": [],
    }
    with connect(config) as conn, conn.cursor() as cur:
        # Renames first, so a later merge can reference the new name.
        for old_name, new_name in spec.renames:
            entity_id = resolve_entity(cur, old_name)
            if entity_id is None:
                summary["missing"].append({"rename": old_name})
                continue
            if not dry_run:
                _rename(cur, entity_id, new_name)
            summary["renamed"].append([old_name, new_name])

        for winner_name, loser_name in spec.merges:
            winner = resolve_entity(cur, winner_name)
            loser = resolve_entity(cur, loser_name)
            if winner is None or loser is None:
                summary["missing"].append(
                    {"merge": [winner_name, loser_name], "winner_id": winner, "loser_id": loser}
                )
                continue
            if dry_run:
                summary["merged"].append([winner_name, loser_name, "would merge"])
                continue
            summary["merged"].append([winner_name, loser_name, _merge(cur, winner, loser)])

        for name in spec.hide:
            entity_id = resolve_entity(cur, name)
            if entity_id is None:
                summary["missing"].append({"hide": name})
                continue
            if not dry_run:
                _set_hidden(cur, entity_id, True)
            summary["hidden"].append(name)

        for name in spec.unhide:
            entity_id = resolve_entity(cur, name)
            if entity_id is None:
                summary["missing"].append({"unhide": name})
                continue
            if not dry_run:
                _set_hidden(cur, entity_id, False)
            summary["unhidden"].append(name)
    return summary
