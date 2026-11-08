"""Convert Re-DocRED (Wikipedia, Wikidata-labelled) into a gold set and a schema-coverage report.

Re-DocRED labels each fact with a Wikidata property; the relation registry
(``RELATION_REGISTRY``, D40/D41) lists each relation's Wikidata properties,
``^`` marking an inverse. A fact is *in schema* when its property maps to a
registry relation and its two entities' types fit that relation's head and
tail. The report counts how much of the data the registry can express, which
needs no model at all (D42).

The article text is CC BY-SA (Wikipedia): write it and the gold file outside
this repository.

    python -I scripts/redocred_gold.py --input ../redocred/raw/test_revised.json \\
        --out-text ../redocred/data/sample50.md --out-gold ../redocred/gold/sample50.jsonl \\
        --sample 50 --seed 0 [--modules core]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from entity_graph.extract.ontology import RELATION_REGISTRY, stored_relation

# Re-DocRED entity types -> ontology labels they can stand for. MISC mixes
# works, products, events and natural features (creeks, buildings), so it is
# not read as an event: as one, every "creek located in county" fact counted
# as an event's place. Event relations are undercounted instead. NUM is never
# a network entity.
TYPE_LABELS = {
    "PER": {"person"}, "ORG": {"organization"}, "LOC": {"location"},
    "MISC": {"product", "concept"}, "TIME": {"date"}, "NUM": set(),
}
ACTOR = {"PER", "ORG"}


def detok(tokens: list[str]) -> str:
    """Tokens back to text; the same function builds the text and every quote."""
    text = " ".join(tokens)
    text = re.sub(r" ([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    text = re.sub(r" ('s|'re|'ve|'ll|'d|n't|')(?=\s|$)", r"\1", text)
    return text.replace("`` ", '"').replace(" ''", '"')


def wikidata_index(modules: set[str] | None) -> dict[str, list[tuple]]:
    """Wikidata property -> [(spec, inverse)] for the registry (or some modules)."""
    index: dict[str, list[tuple]] = collections.defaultdict(list)
    for spec in RELATION_REGISTRY:
        if modules and spec.module not in modules:
            continue
        for standard, term in spec.maps:
            if standard == "Wikidata":
                index[term.lstrip("^")].append((spec, term.startswith("^")))
    return index


def entity_type(vertex: list[dict]) -> str:
    return collections.Counter(m["type"] for m in vertex).most_common(1)[0][0]


def kind(h: str, t: str) -> str:
    """Coarse kind of a fact, by its entity types, for the report."""
    if {h, t} & {"TIME", "NUM"}:
        return "date/number"
    if h in ACTOR and t in ACTOR:
        return "actor-actor"
    if (h in ACTOR and t == "LOC") or (t in ACTOR and h == "LOC"):
        return "actor-place"
    if h == "LOC" and t == "LOC":
        return "place-place"
    return "work/event (MISC)"


def convert(doc: dict, index: dict, doc_name: str) -> tuple[str, list[dict], list[tuple]]:
    """The article as text, its in-schema gold facts, and (kind, pid, in_schema) per label."""
    sents = [detok(s) for s in doc["sents"]]
    vertices = doc["vertexSet"]
    types = [entity_type(v) for v in vertices]
    aliases = [sorted({detok(doc["sents"][m["sent_id"]][m["pos"][0]:m["pos"][1]]) for m in v},
                      key=len, reverse=True) for v in vertices]
    canonical = [a[0] for a in aliases]
    facts, tally, seen = [], [], set()
    for label in doc["labels"]:
        h, t, pid = label["h"], label["t"], label["r"]
        fitted = []
        for spec, inverse in index.get(pid, ()):
            head, tail = (t, h) if inverse else (h, t)
            if TYPE_LABELS[types[head]] & set(spec.head) and TYPE_LABELS[types[tail]] & set(spec.tail):
                fitted.append((spec, head, tail, inverse))
        tally.append((kind(types[h], types[t]), pid, bool(fitted)))
        for spec, head, tail, inverse in fitted:
            key = (head, spec.name, tail)
            if key in seen:
                continue
            seen.add(key)
            ids = label.get("evidence") or sorted(
                {m["sent_id"] for m in vertices[h]} | {m["sent_id"] for m in vertices[t]})
            evidence = " ".join(sents[min(ids):max(ids) + 1])
            facts.append({
                "doc": doc_name, "subject": canonical[head], "subject_aliases": aliases[head],
                "relation": stored_relation(spec.name), "object": canonical[tail],
                "object_aliases": aliases[tail], "modality": "fact", "role": "fact",
                "evidence": evidence,
                "note": f"Re-DocRED {'^' if inverse else ''}{pid}: {doc['title']}",
            })
    return f"# {doc['title']}\n\n" + " ".join(sents), facts, tally


def report(tally: list[tuple], label: str) -> str:
    lines = [f"Schema coverage ({label}): {sum(1 for *_, ok in tally if ok)}/{len(tally)} "
             f"Re-DocRED facts in schema"]
    by_kind = collections.defaultdict(list)
    for k, pid, ok in tally:
        by_kind[k].append((pid, ok))
    for k in ("actor-actor", "actor-place", "place-place", "work/event (MISC)", "date/number"):
        rows = by_kind.get(k, [])
        if not rows:
            continue
        inside = sum(ok for _, ok in rows)
        missing = collections.Counter(pid for pid, ok in rows if not ok).most_common(8)
        lines.append(f"  {k}: {inside}/{len(rows)} in schema; top outside: "
                     + ", ".join(f"{pid} x{n}" for pid, n in missing))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-text")
    parser.add_argument("--out-gold")
    parser.add_argument("--sample", type=int, default=0, help="articles to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--modules", default="", help="comma-separated schema modules (default all)")
    args = parser.parse_args()

    docs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    modules = {m.strip() for m in args.modules.split(",") if m.strip()} or None
    index = wikidata_index(modules)
    label = ", ".join(sorted(modules)) if modules else "whole registry"

    every = [t for doc in docs for t in convert(doc, index, "x")[2]]
    print(f"Re-DocRED: {len(docs)} articles, {len(every)} facts")
    print(report(every, label))

    if not (args.out_text and args.out_gold):
        return
    chosen = sorted(random.Random(args.seed).sample(range(len(docs)), args.sample)
                    if args.sample else range(len(docs)))
    doc_name = Path(args.out_text).name
    texts, gold, tally = [], [], []
    for i in chosen:
        text, facts, rows = convert(docs[i], index, doc_name)
        texts.append(text)
        gold.extend(facts)
        tally.extend(rows)
    full = "\n\n".join(texts) + "\n"
    flat = " ".join(full.split())
    for fact in gold:
        assert " ".join(fact["evidence"].split()) in flat, fact["evidence"][:80]
    for path in (args.out_text, args.out_gold):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_text).write_text(full, encoding="utf-8")
    Path(args.out_gold).write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in gold), encoding="utf-8")
    print(f"Sample: {len(chosen)} articles, {len(tally)} facts, {len(gold)} gold facts in schema "
          f"-> {args.out_gold}")
    print(collections.Counter(f["relation"] for f in gold).most_common())


if __name__ == "__main__":
    main()
