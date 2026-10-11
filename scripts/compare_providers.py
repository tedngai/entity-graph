"""Compare two SystemOne/Jev providers on the gold fact set and alignment pairs.

Read-only: writes nothing to Postgres and never tunes the frozen threshold.
Both providers are driven through the same ``SystemOneClient``, so any
difference is the model, not our parsing.

Example (Pratt Jev vs a local Clef-Flash server):

    python scripts/compare_providers.py \
        --a-name jev  --a-url https://llm.pratt.edu/v1/systemone --a-model pratt-jev \
        --b-name clef --b-url http://192.168.1.242:8001/v1/systemone \
        --b-model Cloudflare/clef-flash
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entity_graph.config import Config, load_config  # noqa: E402
from entity_graph.eval_facts import GoldFact, load_gold  # noqa: E402
from entity_graph.extract.gate import should_promote  # noqa: E402
from entity_graph.extract.jev import (  # noqa: E402
    RELATION_VALIDATION_THRESHOLD,
    SystemOneClient,
    align_pair,
    judge_relations,
)

# Alignment pairs with an intended outcome. "merge" pairs are canonical +
# alias; "separate" pairs are unrelated entities of the same domain.
ALIGNMENT_PAIRS: list[tuple[str, str, str, str]] = [
    ("Maria Chen", "PERSON", "Dr. Maria Chen", "merge"),
    ("GreenGrid Energy", "ORG", "GreenGrid", "merge"),
    ("Tobias Renner", "PERSON", "Renner", "merge"),
    ("Nordwind Capital", "ORG", "Nordwind", "merge"),
    ("GreenGrid Energy", "ORG", "Nordwind Capital", "separate"),
    ("Maria Chen", "PERSON", "Tobias Renner", "separate"),
    ("Berlin", "LOCATION", "Hamburg", "LOCATION"),
]


def make_client(name: str, url: str, model: str, key: str) -> SystemOneClient:
    base: Config = load_config()
    cfg = replace(base, systemone_url=url, systemone_model=model, systemone_api_key=key)
    return SystemOneClient(cfg)


def _expected_promote(fact: GoldFact) -> bool:
    return fact.role == "fact"


def compare_relations(name: str, client: SystemOneClient, gold: list[GoldFact]) -> dict:
    claims = [
        {
            "id": i,
            "subject": fact.subject,
            "relation": fact.relation,
            "object": fact.object,
            "evidence": fact.evidence,
        }
        for i, fact in enumerate(gold)
    ]
    started = time.perf_counter()
    try:
        verdicts = judge_relations(client, claims)
    except Exception as exc:  # noqa: BLE001 - a comparison tool must not crash
        print(f"[{name}] relation judgment failed: {exc}")
        verdicts = {}
    elapsed = time.perf_counter() - started

    rows = []
    correct = leaks = missed = scored = 0
    for i, fact in enumerate(gold):
        judgment = verdicts.get(i)
        if judgment is None:
            rows.append((fact, None, None, None, False))
            continue
        promoted = should_promote(judgment, threshold=RELATION_VALIDATION_THRESHOLD)
        expected = _expected_promote(fact)
        ok = promoted == expected
        scored += 1
        correct += ok
        if promoted and not expected:
            leaks += 1
        if not promoted and expected:
            missed += 1
        rows.append((fact, judgment.assertion, judgment.modality, promoted, ok))
    return {
        "rows": rows,
        "scored": scored,
        "correct": correct,
        "leaks": leaks,
        "missed": missed,
        "elapsed": elapsed,
        "verdicts": verdicts,
    }


def compare_alignment(name: str, client: SystemOneClient) -> dict:
    started = time.perf_counter()
    outcomes = []
    for a, kind_a, b, expected in ALIGNMENT_PAIRS:
        verdict = align_pair(client, {"name": a, "kind": kind_a}, {"name": b, "kind": kind_a})
        outcomes.append((a, b, expected, verdict.outcome if verdict else None,
                         verdict.score if verdict else None))
    return {"outcomes": outcomes, "elapsed": time.perf_counter() - started}


def _fmt(value: object, width: int) -> str:
    text = "" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value))
    return text.ljust(width)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/gold/greengrid_partnership.jsonl")
    parser.add_argument("--a-name", default="jev")
    parser.add_argument("--a-url", default="https://llm.pratt.edu/v1/systemone")
    parser.add_argument("--a-model", default="pratt-jev")
    parser.add_argument("--a-key", default="", help="defaults to the configured Jev key")
    parser.add_argument("--b-name", default="clef")
    parser.add_argument("--b-url", default="http://192.168.1.242:8001/v1/systemone")
    parser.add_argument("--b-model", default="Cloudflare/clef-flash")
    parser.add_argument("--b-key", default="local", help="local servers accept any non-empty key")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    base = load_config()
    a_key = args.a_key or base.systemone_api_key
    a = make_client(args.a_name, args.a_url, args.a_model, a_key)
    b = make_client(args.b_name, args.b_url, args.b_model, args.b_key)

    print(f"A = {args.a_name:6} {args.a_model:28} {args.a_url}  (key {'set' if a.enabled else 'MISSING'})")
    print(f"B = {args.b_name:6} {args.b_model:28} {args.b_url}  (key {'set' if b.enabled else 'MISSING'})")
    print(f"Gold: {len(gold)} items; threshold frozen at {RELATION_VALIDATION_THRESHOLD}\n")

    rel_a = compare_relations(args.a_name, a, gold)
    rel_b = compare_relations(args.b_name, b, gold)

    print("=== Relation gate (does the evidence assert the claim? modality?) ===")
    header = (f"{'role':8} {'subject':22} {'relation':14} {'object':20} | "
              f"{args.a_name[:5]:>6} {'mod':10} {'ok':3} | {args.b_name[:5]:>6} {'mod':10} {'ok':3}")
    print(header)
    print("-" * len(header))
    for (fact, na, ma, pa, oka), (_, nb, mb, pb, okb) in zip(rel_a["rows"], rel_b["rows"]):
        print(
            f"{fact.role:8} {fact.subject[:22]:22} {fact.relation[:14]:14} {fact.object[:20]:20} | "
            f"{_fmt(na, 6)} {_fmt(ma, 10)} {_fmt('Y' if oka else 'n', 3)} | "
            f"{_fmt(nb, 6)} {_fmt(mb, 10)} {_fmt('Y' if okb else 'n', 3)}"
        )

    for name, rel in ((args.a_name, rel_a), (args.b_name, rel_b)):
        scored = rel["scored"]
        acc = rel["correct"] / scored if scored else 0.0
        print(f"\n[{name}] scored {scored}/{len(gold)}  gate-accuracy {acc:.3f}  "
              f"leaks(nonfact/trap promoted) {rel['leaks']}  missed-facts {rel['missed']}  "
              f"time {rel['elapsed']:.2f}s")

    # Agreement between the two providers on the promote decision.
    agree = both = 0
    for (_, _, _, pa, _), (_, _, _, pb, _) in zip(rel_a["rows"], rel_b["rows"]):
        if pa is None or pb is None:
            continue
        both += 1
        agree += pa == pb
    if both:
        print(f"\nPromote-decision agreement: {agree}/{both} = {agree / both:.3f}")

    print("\n=== Entity alignment (Score -> separate / curator / merge) ===")
    al_a = compare_alignment(args.a_name, a)
    al_b = compare_alignment(args.b_name, b)
    print(f"{'a':16} {'b':16} {'intended':10} | {args.a_name[:5]:>7} {'score':>6} | {args.b_name[:5]:>7} {'score':>6}")
    for (a1, b1, exp, oa, sa), (_, _, _, ob, sb) in zip(al_a["outcomes"], al_b["outcomes"]):
        print(f"{a1[:16]:16} {b1[:16]:16} {exp:10} | {_fmt(oa, 7)} {_fmt(sa, 6)} | {_fmt(ob, 7)} {_fmt(sb, 6)}")
    for name, al in ((args.a_name, al_a), (args.b_name, al_b)):
        hit = sum(1 for _, _, exp, out, _ in al["outcomes"] if out == exp)
        print(f"[{name}] alignment intended-match {hit}/{len(al['outcomes'])}  time {al['elapsed']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
