"""Pure promotion rule for relation proposals.

Jev and GLiNER2.5-Decide both produce a ``RelationJudgment``. This module
decides whether that judgment promotes the proposal. It does not call a model
and it does not own the frozen assertion cutoff — the caller passes that in
so a gold-set sweep can vary it without writing the constant.
"""

from __future__ import annotations

from dataclasses import dataclass

# Argmax labels. ``not stated`` from Decide is normalized to ``not_stated``.
FACT = "fact"
MODALITY_LABELS: tuple[str, ...] = (
    "fact",
    "possibility",
    "denial",
    "forecast",
    "not stated",
)


def normalize_modality(label: str | None) -> str | None:
    """Collapse label spelling. Returns None when the judge did not answer."""
    if label is None:
        return None
    text = str(label).strip().lower().replace(" ", "_")
    if not text:
        return None
    if text in {"notstated", "not_stated", "none", "no_relation"}:
        return "not_stated"
    return text


@dataclass
class RelationJudgment:
    """One gate reading of a candidate edge.

    ``assertion`` is a Jev Noul in [0, 1], or None when that question was not
    asked. ``modality`` is an argmax label, never a thresholded probability.
    ``agrees`` is the Decide relation-agreement check; None means it was not
    run (unknown endpoint types, or the Jev path, which uses Noul instead).
    """

    assertion: float | None = None
    modality: str | None = None
    agrees: bool | None = None
    source: str = "none"  # jev | decide | none


def should_promote(judgment: RelationJudgment | None, *, threshold: float) -> bool:
    """Return True when the proposal should become a fact.

    Unscored proposals promote. That is the offline degrade path: no Jev key
    and no local Decide model. A scored proposal is rejected when:

    * modality was answered and the argmax is not ``fact`` (no probability cutoff)
    * Decide disagreed with the proposed relation
    * the source is Jev and the assertion Noul is below ``threshold``

    The Decide path does not apply ``threshold``. Decide probabilities are not
    Jev Nouls, and the modality decision is an argmax.
    """
    if judgment is None or judgment.source in ("", "none"):
        return True
    modality = normalize_modality(judgment.modality)
    if modality is not None and modality != FACT:
        return False
    if judgment.agrees is False:
        return False
    if (
        judgment.source == "jev"
        and judgment.assertion is not None
        and judgment.assertion < threshold
    ):
        return False
    return True
