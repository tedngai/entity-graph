"""Score predicted triples against a gold fact set.

The scorer does not tune ``RELATION_VALIDATION_THRESHOLD``. A sweep is printed
so a later decision record can justify a change; this module never writes the
constant. See DECISIONS.md D15.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .extract.gate import RelationJudgment, normalize_modality, should_promote
from .extract.ontology import stored_relation

# Informational only. The live cutoff stays in extract/jev.py.
SWEEP_THRESHOLDS: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass(frozen=True)
class GoldFact:
    doc: str
    subject: str
    subject_aliases: tuple[str, ...]
    relation: str
    object: str
    object_aliases: tuple[str, ...]
    modality: str
    role: str  # fact | nonfact | trap
    evidence: str = ""
    note: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.subject, stored_relation(self.relation), self.object)


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    matched: int
    predicted: int
    gold: int
    trap_leaks: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "f1": round(self.f1, 3),
            "matched": self.matched,
            "predicted": self.predicted,
            "gold": self.gold,
            "trap_leaks": self.trap_leaks,
        }


def _norm(text: str) -> str:
    return " ".join(text.casefold().replace("_", " ").split())


def _names(canonical: str, aliases: tuple[str, ...]) -> set[str]:
    return {_norm(canonical), *(_norm(alias) for alias in aliases if alias)}


def _doc_ok(prediction: dict, gold: GoldFact) -> bool:
    raw = prediction.get("doc") or ""
    if not raw:
        return True
    return Path(str(raw)).name == Path(gold.doc).name


def matches(prediction: dict, gold: GoldFact) -> bool:
    """Alias-aware, direction-sensitive match. Relation spelling is normalized."""
    if not _doc_ok(prediction, gold):
        return False
    subject = _norm(str(prediction.get("subject") or prediction.get("source") or ""))
    obj = _norm(str(prediction.get("object") or prediction.get("target") or ""))
    relation = stored_relation(str(prediction.get("relation") or ""))
    return (
        subject in _names(gold.subject, gold.subject_aliases)
        and obj in _names(gold.object, gold.object_aliases)
        and relation == stored_relation(gold.relation)
    )


def load_gold(path: str | Path) -> list[GoldFact]:
    """Load a JSONL gold file. One object per line; blank lines ignored."""
    facts: list[GoldFact] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        role = str(row.get("role") or "").strip().lower()
        if role not in {"fact", "nonfact", "trap"}:
            raise ValueError(f"{path}:{line_no}: role must be fact, nonfact, or trap")
        facts.append(
            GoldFact(
                doc=str(row.get("doc") or ""),
                subject=str(row["subject"]),
                subject_aliases=tuple(row.get("subject_aliases") or ()),
                relation=stored_relation(str(row["relation"])),
                object=str(row["object"]),
                object_aliases=tuple(row.get("object_aliases") or ()),
                modality=normalize_modality(str(row.get("modality") or "")) or "",
                role=role,
                evidence=str(row.get("evidence") or ""),
                note=str(row.get("note") or ""),
            )
        )
    return facts


def evidence_missing(gold: list[GoldFact], text: str) -> list[str]:
    """Evidence quotes that are not substrings of the source document."""
    return [fact.evidence for fact in gold if fact.evidence and fact.evidence not in text]


def judgment_from_prediction(prediction: dict) -> RelationJudgment | None:
    """Build a judgment when the prediction carries gate fields. Else None."""
    has_gate = any(prediction.get(key) is not None for key in ("assertion", "modality", "agrees"))
    if not has_gate and not prediction.get("source"):
        return None
    agrees = prediction.get("agrees")
    if agrees is not None:
        agrees = bool(agrees)
    source = str(prediction.get("source") or "")
    if not source:
        source = "jev" if prediction.get("assertion") is not None else "decide"
    assertion = prediction.get("assertion")
    return RelationJudgment(
        assertion=float(assertion) if assertion is not None else None,
        modality=normalize_modality(prediction.get("modality")),
        agrees=agrees,
        source=source,
    )


def _prf(matched: int, predicted: int, gold: int, trap_leaks: int = 0) -> Score:
    precision = matched / predicted if predicted else 0.0
    recall = matched / gold if gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return Score(precision, recall, f1, matched, predicted, gold, trap_leaks)


def _count(predictions: list[dict], targets: list[GoldFact], traps: list[GoldFact]) -> Score:
    matched = sum(1 for fact in targets if any(matches(pred, fact) for pred in predictions))
    leaks = sum(1 for pred in predictions if any(matches(pred, trap) for trap in traps))
    return _prf(matched, len(predictions), len(targets), leaks)


def promote_subset(predictions: list[dict], *, threshold: float) -> list[dict]:
    """Apply the live promotion rule. Predictions with no gate fields are kept."""
    kept: list[dict] = []
    for prediction in predictions:
        judgment = judgment_from_prediction(prediction)
        if should_promote(judgment, threshold=threshold):
            kept.append(prediction)
    return kept


def score_predictions(
    gold: list[GoldFact],
    predictions: list[dict],
    *,
    threshold: float,
) -> dict:
    """Score extraction (right label, any modality) and the fact gate.

    Extraction gold is ``fact`` plus ``nonfact`` (the relation is in the text).
    Fact-store gold is ``fact`` only. A promoted trap or nonfact is a leak,
    not a match. ``threshold`` is read, never written.
    """
    extract_gold = [fact for fact in gold if fact.role in {"fact", "nonfact"}]
    fact_gold = [fact for fact in gold if fact.role == "fact"]
    traps = [fact for fact in gold if fact.role == "trap"]
    gated = promote_subset(predictions, threshold=threshold)
    return {
        "extraction": _count(predictions, extract_gold, traps).as_dict(),
        "fact_gate": _count(gated, fact_gold, traps).as_dict(),
        "threshold": threshold,
        "threshold_frozen": True,
    }


def threshold_sweep(
    gold: list[GoldFact],
    predictions: list[dict],
    thresholds: tuple[float, ...] = SWEEP_THRESHOLDS,
) -> list[dict]:
    """Informational curve. Does not change the frozen cutoff."""
    rows = []
    for cutoff in thresholds:
        scored = score_predictions(gold, predictions, threshold=cutoff)
        rows.append({"threshold": cutoff, **scored["fact_gate"]})
    return rows


def load_predictions(path: str | Path) -> list[dict]:
    """Load a JSON list or a JSONL file of predicted triples."""
    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"{path} must be a JSON list of triples")
        return data
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def format_report(
    gold: list[GoldFact],
    scored: dict | None,
    sweep: list[dict] | None,
    *,
    threshold: float = 0.5,
) -> str:
    facts = sum(1 for fact in gold if fact.role == "fact")
    nonfacts = sum(1 for fact in gold if fact.role == "nonfact")
    traps = sum(1 for fact in gold if fact.role == "trap")
    shown = scored["threshold"] if scored else threshold
    lines = [
        f"Gold: {facts} facts, {nonfacts} non-facts, {traps} traps.",
        f"RELATION_VALIDATION_THRESHOLD is frozen at {shown}. This command does not tune it.",
    ]
    if scored is None:
        return "\n".join(lines)
    lines.append(f"Extraction (right relation, before the gate): {scored['extraction']}")
    lines.append(f"Fact gate (promoted vs gold facts): {scored['fact_gate']}")
    if sweep:
        lines.append("Sweep (informational; the constant is unchanged):")
        for row in sweep:
            lines.append(
                f"  {row['threshold']:.1f}  P={row['precision']:.3f} "
                f"R={row['recall']:.3f} F1={row['f1']:.3f} leaks={row['trap_leaks']}"
            )
    return "\n".join(lines)
