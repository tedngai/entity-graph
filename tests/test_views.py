"""The graph-view contract as finally defined by schema.sql + migrations.

Pure text checks, no database: init-db applies db/schema.sql and then every
migration in order, so the last CREATE OR REPLACE of a view is what runs.
"""

from __future__ import annotations

import re
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "db"


def _final_view(name: str) -> str:
    sources = [DB / "schema.sql", *sorted((DB / "migrations").glob("*.sql"))]
    pattern = re.compile(
        rf"CREATE OR REPLACE VIEW {name} AS(.*?);", re.DOTALL | re.IGNORECASE
    )
    definition = ""
    for path in sources:
        for match in pattern.finditer(path.read_text(encoding="utf-8")):
            definition = match.group(1)
    assert definition, f"{name} is never defined"
    return definition


def test_edges_view_skips_self_loops() -> None:
    assert "r.source_id <> r.target_id" in _final_view("v_graph_edges")


def test_nodes_view_ignores_self_loops_and_curated_entities() -> None:
    view = _final_view("v_graph_nodes")
    assert "r.source_id <> r.target_id" in view
    assert "merged_into IS NULL" in view
    assert "NOT e.is_hidden" in view
