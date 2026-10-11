"""Gold fact set and the scorer. No database, no models."""

from pathlib import Path

from entity_graph.eval_facts import (
    evidence_missing,
    load_gold,
    score_predictions,
    threshold_sweep,
)
from entity_graph.extract.jev import RELATION_VALIDATION_THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold" / "greengrid_partnership.jsonl"
SAMPLE = ROOT / "data" / "samples" / "greengrid_partnership.txt"


def test_gold_evidence_is_in_the_sample() -> None:
    gold = load_gold(GOLD)
    text = SAMPLE.read_text(encoding="utf-8")
    assert evidence_missing(gold, text) == []
    roles = {fact.role for fact in gold}
    assert roles == {"fact", "nonfact", "trap"}
    assert any(fact.relation == "FOUNDED" and fact.role == "fact" for fact in gold)
    assert any(fact.relation == "ACQUIRED" and fact.role == "trap" for fact in gold)


def test_scorer_counts_aliases_direction_and_leaks() -> None:
    gold = load_gold(GOLD)
    predictions = [
        {"subject": "Maria Chen", "relation": "founded", "object": "GreenGrid Energy",
         "modality": "fact", "assertion": 0.9, "source": "jev"},
        {"subject": "GreenGrid Energy", "relation": "FOUNDED", "object": "Dr. Maria Chen",
         "modality": "not_stated", "assertion": 0.1, "source": "jev"},
        {"subject": "GreenGrid Energy", "relation": "ACQUIRED", "object": "Nordwind Capital",
         "modality": "fact", "assertion": 0.8, "source": "jev"},
        {"subject": "Nordwind Capital", "relation": "INVESTED_IN", "object": "GreenGrid Energy",
         "modality": "forecast", "assertion": 0.95, "source": "jev"},
    ]
    scored = score_predictions(gold, predictions, threshold=RELATION_VALIDATION_THRESHOLD)
    assert scored["threshold_frozen"] is True
    assert scored["extraction"]["matched"] >= 2
    assert scored["fact_gate"]["trap_leaks"] == 1
    assert scored["fact_gate"]["matched"] == 1


def test_sweep_does_not_write_the_cutoff() -> None:
    before = RELATION_VALIDATION_THRESHOLD
    gold = load_gold(GOLD)
    rows = threshold_sweep(gold, [{
        "subject": "Tobias Renner",
        "relation": "LEADS",
        "object": "Nordwind Capital",
        "modality": "fact",
        "assertion": 0.42,
        "source": "jev",
    }])
    assert RELATION_VALIDATION_THRESHOLD == before == 0.5
    by_cutoff = {row["threshold"]: row["matched"] for row in rows}
    assert by_cutoff[0.3] == 1
    assert by_cutoff[0.5] == 0
