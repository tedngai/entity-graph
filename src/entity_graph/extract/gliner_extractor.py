"""GLiNER2 extractor backend — the primary extraction path (PROGRESS P4.4).

GLiNER2.5 decodes entities and relations together through JointIE so endpoint
types, uniqueness, and no-self-loop constraints hold in the returned graph.
Independent ``.relations()`` remains only as a fallback when the installed
``gliner2`` has no ``joint_ie`` module. See DECISIONS.md D13.

Verified against gliner2 2.1.0 (2026-10-10) for the independent shape, and
against the GLiNER2.5 JointIE tutorial for the primary path:
  * ``JointIE.from_pretrained`` loads boundary checkpoints (including
    ``fastino/gliner2.5-base-v1``).
  * schema: ``create_schema().entities(...).relation(...).no_self_loops()``
  * ``extract`` returns entities with ids and relations whose head/tail are ids.
Device is chosen from ``GLINER_DEVICE`` (auto | cpu | cuda).
"""

from __future__ import annotations

import logging

from ..chunking import split_sentences
from ..config import load_config
from .base import ExtractionResult, Mention, RelationProposal
from .ontology import (
    ENTITY_DESCRIPTIONS,
    LABEL_TO_TYPE,
    RELATION_SPECS,
    SCHEMA_STEPS,
    relation_kwargs,
    stored_relation,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "fastino/gliner2.5-base-v1"
_INSTALL_HINT = (
    'GLiNER2 is not installed. Run: pip install "gliner2[local]" '
    "(then, for CUDA: pip install torch --index-url "
    "https://download.pytorch.org/whl/cu130)"
)


def _sentence_index(spans: list[tuple[int, int]], position: int) -> int | None:
    for index, (start, end) in enumerate(spans):
        if start <= position < end:
            return index
    return None


def _evidence_for(text: str, head_start: int, tail_start: int) -> str:
    """Evidence quote spanning the head and tail spans.

    The relation is asserted somewhere between the two arguments, so we quote
    from the start of the head's sentence to the end of the tail's sentence
    rather than just the head's line.
    """
    spans = split_sentences(text)
    if not spans:
        return ""
    head_index = _sentence_index(spans, head_start)
    tail_index = _sentence_index(spans, tail_start)
    if head_index is None and tail_index is None:
        return ""
    if head_index is None:
        head_index = tail_index
    if tail_index is None:
        tail_index = head_index
    lo, hi = sorted((head_index, tail_index))
    return text[spans[lo][0]:spans[hi][1]].strip()


def _clamp(value: object, default: float = 0.5) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def _anchor(text: str, surface: str, start: int, end: int) -> tuple[int, int] | None:
    if surface and 0 <= start < end <= len(text) and text[start:end] == surface:
        return start, end
    if surface:
        idx = text.find(surface)
        if idx >= 0:
            return idx, idx + len(surface)
    return None


def build_joint_schema(joint):
    """Compile the shared ontology into a JointIE schema. Reuse the result."""
    schema = joint.create_schema().entities(dict(ENTITY_DESCRIPTIONS))
    for spec in RELATION_SPECS:
        schema = schema.relation(spec.name, **relation_kwargs(spec))
    if "no_self_loops" in SCHEMA_STEPS:
        schema = schema.no_self_loops()
    return schema


def parse_joint_payload(
    payload: dict,
    text: str,
    *,
    feasible: bool = True,
    doc_key: str = "",
) -> ExtractionResult:
    """Turn a JointIE ``to_dict()`` payload into the extraction contract.

    When ``feasible`` is false the decoder could not satisfy the hard
    constraints. Mentions are kept; relations are dropped rather than stored
    as an inconsistent graph.
    """
    by_id: dict[str, tuple[Mention, str, int]] = {}
    mentions: list[Mention] = []
    for ent in payload.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text") or "").strip()
        if not surface:
            continue
        anchored = _anchor(text, surface, int(ent.get("start") or 0), int(ent.get("end") or 0))
        if anchored is None:
            continue
        start, end = anchored
        label = str(ent.get("type") or "").strip().lower()
        mention = Mention(
            surface=surface,
            entity_type=LABEL_TO_TYPE.get(label, "MISC"),
            start=start,
            end=end,
            confidence=_clamp(ent.get("confidence", 0.5)),
        )
        mentions.append(mention)
        ent_id = ent.get("id")
        if ent_id is not None:
            by_id[str(ent_id)] = (mention, label, start)

    relations: list[RelationProposal] = []
    if feasible:
        for rel in payload.get("relations") or []:
            if not isinstance(rel, dict):
                continue
            head_id = str(rel.get("head") or "")
            tail_id = str(rel.get("tail") or "")
            if not head_id or not tail_id or head_id == tail_id:
                continue
            head = by_id.get(head_id)
            tail = by_id.get(tail_id)
            if head is None or tail is None:
                continue
            head_mention, head_label, head_start = head
            tail_mention, tail_label, tail_start = tail
            if head_mention.surface == tail_mention.surface:
                continue
            relation_name = stored_relation(str(rel.get("type") or ""))
            if not relation_name:
                continue
            relations.append(
                RelationProposal(
                    source_surface=head_mention.surface,
                    target_surface=tail_mention.surface,
                    relation=relation_name,
                    evidence=_evidence_for(text, head_start, tail_start),
                    confidence=_clamp(rel.get("confidence", 0.5)),
                    source_type=head_label,
                    target_type=tail_label,
                )
            )
    return ExtractionResult(doc_key=doc_key, mentions=mentions, relations=relations)


class GlinerExtractor:
    name = "gliner2"

    def __init__(
        self,
        model: str | None = None,
        threshold: float | None = None,
        device: str | None = None,
    ) -> None:
        config = load_config()
        self.model_name = model or config.gliner_model or DEFAULT_MODEL
        self.threshold = config.gliner_threshold if threshold is None else threshold
        self.device = device or config.gliner_device or "auto"
        self._mode = "joint"
        self._model = None
        self._schema = None
        self._joint_config = None
        self._load()

    def _map_location(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load(self) -> None:
        try:
            from gliner2.joint_ie import JointIE, JointIEConfig
        except ImportError as exc:
            try:
                from gliner2 import AutoExtractor  # noqa: F401 — presence check
            except ImportError:
                raise RuntimeError(_INSTALL_HINT) from exc
            log.warning(
                "gliner2.joint_ie is unavailable (%s); falling back to independent "
                "relation extraction. Upgrade gliner2 for typed JointIE.",
                exc,
            )
            self.device = self._map_location()
            self._load_independent()
            return

        location = self._map_location()
        self.device = location
        try:
            self._model = JointIE.from_pretrained(self.model_name, map_location=location)
        except TypeError:
            self._model = JointIE.from_pretrained(self.model_name)
        try:
            self._schema = build_joint_schema(self._model)
            self._joint_config = _joint_config(JointIEConfig, self.threshold)
        except (AttributeError, TypeError) as exc:
            log.warning(
                "JointIE schema failed (%s); falling back to independent extraction.",
                exc,
            )
            self._load_independent()
            return
        self._mode = "joint"
        log.info("GLiNER2 JointIE %s loaded on %s", self.model_name, location)

    def _load_independent(self) -> None:
        from gliner2 import AutoExtractor

        kwargs = {"map_location": self.device} if self.device == "cuda" else {}
        self._model = AutoExtractor.from_pretrained(self.model_name, **kwargs)
        self._mode = "independent"
        log.info("GLiNER2 independent %s loaded on %s", self.model_name, self.device)

    def extract(self, text: str, doc_key: str = "") -> ExtractionResult:
        if self._mode == "joint":
            result = self._model.extract(text, self._schema, config=self._joint_config)
            feasible = bool(getattr(result, "feasible", True))
            if not feasible:
                log.warning(
                    "JointIE infeasible for %s; keeping mentions and dropping relations",
                    doc_key or "document",
                )
            payload = result.to_dict() if hasattr(result, "to_dict") else {}
            parsed = parse_joint_payload(payload, text, feasible=feasible, doc_key=doc_key)
        else:
            parsed = self._extract_independent(text, doc_key)
        log.info(
            "GLiNER2 extracted %d mentions, %d relations for %s",
            len(parsed.mentions), len(parsed.relations), doc_key or "document",
        )
        return parsed

    def _extract_independent(self, text: str, doc_key: str) -> ExtractionResult:
        """Legacy path: entities and relations decoded separately, untyped."""
        labels = list(ENTITY_DESCRIPTIONS)
        relation_labels = [spec.name.replace("_", " ") for spec in RELATION_SPECS]
        schema = self._model.create_schema().entities(labels).relations(relation_labels)
        result = self._model.extract(
            text,
            schema,
            threshold=self.threshold,
            include_confidence=True,
            include_spans=True,
        )
        mentions: list[Mention] = []
        for label, items in (result.get("entities") or {}).items():
            entity_type = LABEL_TO_TYPE.get(str(label).lower(), "MISC")
            for item in items or []:
                surface = str(item.get("text", "")).strip()
                if not surface:
                    continue
                mentions.append(
                    Mention(
                        surface=surface,
                        entity_type=entity_type,
                        start=int(item.get("start", 0)),
                        end=int(item.get("end", 0)),
                        confidence=_clamp(item.get("confidence", 0.5)),
                    )
                )
        relations: list[RelationProposal] = []
        for relation, items in (result.get("relation_extraction") or {}).items():
            relation_name = stored_relation(str(relation))
            for item in items or []:
                head = item.get("head") or {}
                tail = item.get("tail") or {}
                source = str(head.get("text", "")).strip()
                target = str(tail.get("text", "")).strip()
                if not (source and target and relation_name) or source == target:
                    continue
                relations.append(
                    RelationProposal(
                        source_surface=source,
                        target_surface=target,
                        relation=relation_name,
                        evidence=_evidence_for(
                            text, int(head.get("start", 0)), int(tail.get("start", 0))
                        ),
                        confidence=_clamp(item.get("confidence", 0.5)),
                    )
                )
        return ExtractionResult(doc_key=doc_key, mentions=mentions, relations=relations)


def _joint_config(config_cls, threshold: float):
    """Build JointIEConfig, dropping kwargs an older gliner2 does not accept."""
    kwargs = {
        "optimizer": "beam",
        "beam_size": 32,
        "include_confidence": True,
        "include_spans": True,
        "entity_threshold": threshold,
    }
    try:
        return config_cls(**kwargs)
    except TypeError:
        return config_cls(optimizer="beam", beam_size=32)
