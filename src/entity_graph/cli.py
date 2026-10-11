"""entity-graph command line interface.

Run from the repo root so db/schema.sql and .env resolve. Heavy imports are
lazy so `entity-graph --help` works with only the base dependencies.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entity-graph",
        description="Extract entities/relationships from unstructured documents, "
                    "store key facts in PostgreSQL, analyze and visualize with NetworkX.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="apply db/schema.sql to the database")

    p_pre = sub.add_parser(
        "preprocess",
        help="clean OCR text and write it out (no database; useful for inspection)",
    )
    p_pre.add_argument("--input", required=True, help="OCR document to clean")
    p_pre.add_argument("--out", required=True, help="path to write the cleaned text")

    p_ingest = sub.add_parser("ingest", help="load documents into the documents table")
    p_ingest.add_argument("--input", default="data/samples", help="directory of documents")

    p_extract = sub.add_parser("extract", help="run the extraction backend over new documents")
    p_extract.add_argument("--force", action="store_true", help="re-extract all documents")
    p_extract.add_argument("--input", default=None,
                           help="ingest this directory first (optional)")
    p_extract.add_argument("--escalate", action="store_true",
                           help="escalate uncertain documents to the LLM (needs an LLM key)")

    sub.add_parser("resolve", help="link unresolved mentions to canonical entities")

    sub.add_parser("curator", help="list uncertain entity pairs queued by Jev alignment")

    p_build = sub.add_parser("build", help="promote relation proposals into relationships + facts")
    p_build.add_argument("--no-validate", action="store_true",
                         help="skip Jev relation validation (promote on extractor confidence)")

    sub.add_parser("analyze", help="NetworkX report (degree, betweenness, communities)")

    p_viz = sub.add_parser("visualize", help="interactive HTML graph via pyvis")
    p_viz.add_argument("--out", default=None, help="output path (default out/graph.html)")

    p_run = sub.add_parser("run", help="full pipeline: ingest -> extract -> resolve -> build")
    p_run.add_argument("--input", default="data/samples", help="directory of documents")
    p_run.add_argument("--no-validate", action="store_true",
                       help="skip Jev relation validation")
    p_run.add_argument("--escalate", action="store_true",
                       help="escalate uncertain documents to the LLM (needs an LLM key)")

    p_eval = sub.add_parser(
        "eval",
        help="score predictions against the gold fact set (does not tune thresholds)",
    )
    p_eval.add_argument(
        "--gold",
        default="data/gold/greengrid_partnership.jsonl",
        help="JSONL gold facts (fact / nonfact / trap)",
    )
    p_eval.add_argument(
        "--predictions",
        default=None,
        help="JSON list or JSONL of {subject, relation, object} triples",
    )
    p_eval.add_argument(
        "--sample",
        default=None,
        help="source text used to check that gold evidence quotes are substrings",
    )
    p_eval.add_argument(
        "--from-db",
        action="store_true",
        help="score current facts already stored in Postgres",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    from .config import load_config
    config = load_config()
    if getattr(args, "escalate", False):
        from dataclasses import replace

        config = replace(config, escalate=True)

    if args.command == "init-db":
        from .db import apply_schema
        apply_schema(config)
        print(f"Schema applied to {config.pg_db}@{config.pg_host}:{config.pg_port}")

    elif args.command == "ingest":
        from .ingest import ingest_directory
        docs = ingest_directory(args.input, config)
        print(f"Ingested {len(docs)} new document(s) from {args.input}")

    elif args.command == "preprocess":
        from .preprocess import clean_ocr_text, looks_like_ocr
        src = Path(args.input)
        raw = src.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_ocr_text(raw) if looks_like_ocr(raw) else raw
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(cleaned, encoding="utf-8")
        detected = "OCR scaffolding detected" if looks_like_ocr(raw) else "no OCR scaffolding"
        print(f"Wrote {out} ({len(raw)} -> {len(cleaned)} chars; {detected})")

    elif args.command == "extract":
        from .ingest import ingest_directory
        from .pipeline import extract_documents
        if args.input:
            ingest_directory(args.input, config)
        count = extract_documents(config, force=args.force)
        print(f"Extracted {count} document(s) with backend {config.extraction_backend!r}")

    elif args.command == "resolve":
        from .resolve import resolve_unresolved
        print(f"Resolved {resolve_unresolved(config)} mention(s)")

    elif args.command == "curator":
        from .resolve import list_curator_queue
        rows = list_curator_queue(config)
        if not rows:
            print("Curator queue is empty.")
        else:
            print(f"{len(rows)} pair(s) awaiting review:")
            for rid, name_a, type_a, name_b, type_b, score, sidecars in rows:
                print(f"  #{rid}  {name_a!r} ({type_a})  ~?~  {name_b!r} ({type_b})  "
                      f"score={score}  {sidecars}")

    elif args.command == "build":
        from .pipeline import build
        built = build(config, validate=not args.no_validate)
        print(f"Built {built} relationship(s) from proposals")

    elif args.command == "analyze":
        from .analyze import analyze_graph, format_report
        from .graph_store import load_graph
        graph = load_graph(config)
        if graph.number_of_nodes() == 0:
            print("Graph is empty — run `entity-graph run --input data/samples` first.")
            return 1
        print(format_report(analyze_graph(graph)))

    elif args.command == "visualize":
        from .graph_store import load_graph
        from .visualize import visualize_graph
        out = args.out or str(config.out_dir / "graph.html")
        path = visualize_graph(load_graph(config), out)
        print(f"Visualization written to {path}")

    elif args.command == "run":
        from .analyze import analyze_graph, format_report
        from .graph_store import load_graph
        from .pipeline import run_pipeline
        summary = run_pipeline(args.input, config, validate=not args.no_validate)
        print("Pipeline summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        graph = load_graph(config)
        if graph.number_of_nodes():
            print()
            print(format_report(analyze_graph(graph)))

    elif args.command == "eval":
        return _eval(args, config)
    return 0


def _eval(args, config) -> int:
    """Validate the gold file and, if predictions are given, score them.

    Never writes RELATION_VALIDATION_THRESHOLD. A sweep is printed so a later
    decision record can justify a change.
    """
    from .eval_facts import (
        evidence_missing,
        format_report,
        load_gold,
        load_predictions,
        score_predictions,
        threshold_sweep,
    )
    from .extract.jev import RELATION_VALIDATION_THRESHOLD

    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"Gold file not found: {gold_path}")
        return 1
    gold = load_gold(gold_path)

    sample = Path(args.sample) if args.sample else None
    if sample is None and gold and gold[0].doc:
        candidate = Path("data/samples") / Path(gold[0].doc).name
        if candidate.exists():
            sample = candidate
    if sample is not None and sample.exists():
        missing = evidence_missing(gold, sample.read_text(encoding="utf-8"))
        if missing:
            print(f"Gold evidence not found in {sample}:")
            for quote in missing:
                print(f"  {quote!r}")
            return 1

    predictions = None
    if args.predictions:
        predictions = load_predictions(args.predictions)
    elif args.from_db:
        predictions = _facts_from_db(config)
        if predictions is None:
            print(format_report(gold, None, None, threshold=RELATION_VALIDATION_THRESHOLD))
            return 1

    if predictions is None:
        print(format_report(gold, None, None, threshold=RELATION_VALIDATION_THRESHOLD))
        if sample is not None and sample.exists():
            print(f"Evidence quotes checked against {sample}.")
        print("Pass --predictions triples.jsonl to score extractor output.")
        return 0

    scored = score_predictions(
        gold, predictions, threshold=RELATION_VALIDATION_THRESHOLD,
    )
    sweep = threshold_sweep(gold, predictions) if any(
        row.get("assertion") is not None for row in predictions
    ) else None
    print(format_report(gold, scored, sweep, threshold=RELATION_VALIDATION_THRESHOLD))
    return 0


def _facts_from_db(config) -> list[dict] | None:
    """Current stored facts as prediction rows. None if Postgres is unreachable."""
    from .db import connect

    try:
        with connect(config) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT se.canonical_name, r.relation, te.canonical_name, d.doc_key
                FROM facts f
                JOIN relationships r ON r.id = f.relationship_id
                JOIN entities se ON se.id = r.source_id
                JOIN entities te ON te.id = r.target_id
                LEFT JOIN documents d ON d.id = f.document_id
                WHERE f.is_current
                """
            )
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - offline eval must not require a database
        print(f"Could not read facts from Postgres ({exc}).")
        return None
    return [
        {
            "subject": subject,
            "relation": relation,
            "object": obj,
            "doc": Path(path).name if path else "",
        }
        for subject, relation, obj, path in rows
    ]


if __name__ == "__main__":
    raise SystemExit(main())
