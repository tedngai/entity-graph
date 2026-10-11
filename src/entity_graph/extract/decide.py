"""Local relation gate using GLiNER2.5-Decide.

Runs only when Jev is not configured. Modality and relation agreement are
separate ``classify`` calls: tasks that share a prompt shift each other's
answers. Modality uses short labels and an argmax, not a probability cutoff.
Relation agreement is asked only when the ontology knows both endpoint types.
See DECISIONS.md D14 and https://lyonwj.com/blog/gliner-decide-knowledge-graph-extraction.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

from .gate import MODALITY_LABELS, RelationJudgment, normalize_modality
from .ontology import display_relation, legal_display_labels, stored_relation

log = logging.getLogger(__name__)

NO_RELATION = "no relation"
# (text, task, labels, instruction) -> (label, confidence)
ClassifyFn = Callable[..., tuple[str | None, float | None]]

_CLASSIFIERS: dict[tuple[str, str], ClassifyFn] = {}
_PAREN = re.compile(r"[()\[\]]")


def judge_relations(
    claims: list[dict],
    *,
    model: str = "fastino/GLiNER2.5-Decide",
    device: str = "auto",
    classify: ClassifyFn | None = None,
) -> dict[int, RelationJudgment]:
    """Score claims with Decide. ``classify`` is injectable for tests.

    A claim the model cannot score is omitted. The promotion rule treats a
    missing judgment as unscored (promote on extractor confidence), not as a
    rejection.
    """
    if not claims:
        return {}
    if classify is None:
        classify = _load_classifier(model, device)
        if classify is None:
            return {}

    verdicts: dict[int, RelationJudgment] = {}
    for claim in claims:
        evidence = str(claim.get("evidence") or "").strip()
        if not evidence:
            continue
        try:
            modality, confidence = classify(evidence, "modality", list(MODALITY_LABELS))
        except Exception as exc:  # noqa: BLE001 - one bad window must not abort the build
            log.warning("Decide modality failed for claim %s (%s)", claim.get("id"), exc)
            continue
        agrees = _relation_agrees(classify, claim, evidence)
        verdicts[int(claim["id"])] = RelationJudgment(
            assertion=confidence,
            modality=normalize_modality(modality),
            agrees=agrees,
            source="decide",
        )
    return verdicts


def _relation_agrees(classify: ClassifyFn, claim: dict, evidence: str) -> bool | None:
    """Argmax agreement with the proposed relation, or None if types are unknown."""
    labels = [_clean(label) for label in legal_display_labels(
        claim.get("source_type"), claim.get("target_type")
    )]
    labels = [label for label in labels if label]
    if not labels:
        return None
    subject = _clean(str(claim.get("subject") or "the subject"))
    obj = _clean(str(claim.get("object") or "the object"))
    instruction = f"How is {subject} related to {obj}?"
    try:
        picked, _confidence = classify(
            evidence, "relation", labels + [NO_RELATION], instruction,
        )
    except TypeError:
        picked, _confidence = classify(evidence, "relation", labels + [NO_RELATION])
    except Exception as exc:  # noqa: BLE001
        log.warning("Decide relation check failed for claim %s (%s)", claim.get("id"), exc)
        return None
    if not picked:
        return None
    if display_relation(picked) == NO_RELATION:
        return False
    return stored_relation(picked) == stored_relation(str(claim.get("relation") or ""))


def _clean(text: str) -> str:
    """Drop characters the classifier rejects in labels and instructions."""
    return " ".join(_PAREN.sub(" ", text).split())


def _load_classifier(model: str, device: str) -> ClassifyFn | None:
    key = (model, device)
    cached = _CLASSIFIERS.get(key)
    if cached is not None:
        return cached

    location = _resolve_device(device)
    classify = _load_constrained(model, location) or _load_quick(model, location)
    if classify is None:
        return None
    _CLASSIFIERS[key] = classify
    return classify


def _load_constrained(model: str, location: str) -> ClassifyFn | None:
    """Preferred path: ``Classifier`` keeps a probability for every label."""
    try:
        from gliner2.classification import ClassificationSchema, Classifier
    except ImportError:
        return None
    try:
        clf = Classifier.from_pretrained(model, map_location=location)
    except TypeError:
        try:
            clf = Classifier.from_pretrained(model)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load %s (%s); local gate skipped.", model, exc)
            return None
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load %s (%s); local gate skipped.", model, exc)
        return None

    def classify(
        text: str,
        task: str,
        labels: list[str],
        instruction: str | None = None,
    ) -> tuple[str | None, float | None]:
        # Modality is short labels and no instruction. An instruction on that
        # head shifts the argmax (Lyon, 2026-09-24). Relation may name the pair.
        kwargs = {"instruction": instruction} if instruction and task != "modality" else {}
        schema = ClassificationSchema().single(task, labels, **kwargs)
        result = clf.classify(text, schema)
        label = result.value(task) if hasattr(result, "value") else None
        confidence = result.confidence(task) if hasattr(result, "confidence") else None
        if isinstance(label, (list, tuple)):
            label = label[0] if label else None
        if confidence is not None:
            confidence = float(confidence)
        return (str(label) if label else None), confidence

    log.info("GLiNER2.5-Decide %s loaded on %s", model, location)
    return classify


def _load_quick(model: str, location: str) -> ClassifyFn | None:
    """Fallback when this gliner2 has classify_text but no Classifier class."""
    try:
        from gliner2 import AutoExtractor
    except ImportError:
        log.warning(
            "GLiNER2.5-Decide is not installed; local gate skipped. "
            'Run: pip install "gliner2[local]"'
        )
        return None
    try:
        extractor = AutoExtractor.from_pretrained(model, map_location=location)
    except TypeError:
        extractor = AutoExtractor.from_pretrained(model)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load %s (%s); local gate skipped.", model, exc)
        return None

    def classify(
        text: str,
        task: str,
        labels: list[str],
        instruction: str | None = None,
    ) -> tuple[str | None, float | None]:
        del instruction  # classify_text has no per-task instruction argument
        result = extractor.classify_text(text, {task: labels})
        raw = result.get(task) if isinstance(result, dict) else None
        confidence = None
        if isinstance(raw, dict):
            confidence = raw.get("confidence")
            raw = raw.get("label") or raw.get("value")
        if isinstance(raw, list):
            raw = raw[0] if raw else None
        if confidence is not None:
            confidence = float(confidence)
        return (str(raw) if raw else None), confidence

    log.info("GLiNER2.5-Decide %s loaded via classify_text on %s", model, location)
    return classify


def _resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
