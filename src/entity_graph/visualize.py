"""Interactive visualization via pyvis (optional [viz] extra)."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

TYPE_COLORS = {
    "PERSON": "#4e79a7",
    "ORG": "#f28e2b",
    "PLACE": "#59a14f",
    "CONCEPT": "#b07aa1",
    "EVENT": "#e15759",
    "DATE": "#9c755f",
    "MISC": "#bab0ac",
}


def visualize_graph(graph, out_path: str | Path = "out/graph.html") -> Path:
    try:
        from pyvis.network import Network  # lazy: optional extra
    except ImportError as exc:
        raise RuntimeError(
            "pyvis is not installed. Run: pip install -e .[viz]"
        ) from exc

    net = Network(height="850px", width="100%", directed=True, bgcolor="#ffffff")
    for node_id, data in graph.nodes(data=True):
        entity_type = data.get("type", "MISC")
        title = f"{data.get('name')}<br>type: {entity_type}"
        if data.get("description"):
            title += f"<br>{data['description']}"
        net.add_node(
            node_id,
            label=str(data.get("name", node_id)),
            color=TYPE_COLORS.get(entity_type, TYPE_COLORS["MISC"]),
            title=title,
        )
    for u, v, key, data in graph.edges(keys=True, data=True):
        label = key if isinstance(key, str) else str(data.get("relation", ""))
        net.add_edge(u, v, label=label, title=label)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out))
    log.info("Visualization written to %s", out)
    return out
