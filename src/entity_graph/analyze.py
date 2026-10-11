"""NetworkX analysis: structure, centrality, and community detection."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _name(graph, node) -> str:
    return graph.nodes[node].get("name", str(node))


def _top(scores: dict, n: int, graph) -> list[tuple[str, float]]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
    return [(_name(graph, node), round(score, 4)) for node, score in ranked]


def analyze_graph(graph, top_n: int = 10) -> dict:
    """Compute headline metrics. Centrality/community work on the undirected
    projection (direction matters for storage, not for these summaries)."""
    undirected = graph.to_undirected()
    results: dict = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx_density(graph), 4),
        "top_degree": _top(dict(graph.degree()), top_n, graph),
    }
    try:
        results["top_betweenness"] = _top(nx_betweenness(undirected), top_n, graph)
    except Exception as exc:  # pragma: no cover - pathological graphs
        log.warning("Betweenness skipped: %s", exc)
    try:
        results["communities"] = [
            sorted(_name(graph, node) for node in community)
            for community in nx_louvain(undirected)
        ][:top_n]
    except Exception as exc:  # pragma: no cover
        log.warning("Community detection skipped: %s", exc)
    return results


def nx_density(graph) -> float:
    import networkx as nx

    return nx.density(graph)


def nx_betweenness(graph) -> dict:
    import networkx as nx

    return nx.betweenness_centrality(graph)


def nx_louvain(graph) -> list[set]:
    import networkx as nx

    return nx.community.louvain_communities(graph, seed=42)


def format_report(results: dict) -> str:
    lines = [
        "=== entity-graph network report ===",
        f"nodes: {results['nodes']}  edges: {results['edges']}  density: {results['density']}",
        "",
        "Most connected (degree):",
    ]
    for name, score in results.get("top_degree", []):
        lines.append(f"  {score:>4}  {name}")
    if results.get("top_betweenness"):
        lines += ["", "Key bridges (betweenness):"]
        for name, score in results["top_betweenness"]:
            lines.append(f"  {score:>8.4f}  {name}")
    if results.get("communities"):
        lines += ["", f"Communities (louvain, showing {len(results['communities'])}):"]
        for i, community in enumerate(results["communities"], 1):
            preview = ", ".join(community[:8]) + (" …" if len(community) > 8 else "")
            lines.append(f"  [{i}] ({len(community)}) {preview}")
    return "\n".join(lines)
