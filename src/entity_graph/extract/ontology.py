"""Shared entity and relation schema for extraction and the local gate.

JointIE and GLiNER2.5-Decide must agree on which relations are legal between
which types. One definition here; both call sites read it. See DECISIONS.md D13.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ontology labels passed to GLiNER2.5. Descriptions are part of the question.
ENTITY_DESCRIPTIONS: dict[str, str] = {
    "person": "Named people",
    "organization": "Companies, agencies, universities, or investment firms",
    "location": "Cities, countries, regions, or seas",
    "event": "Named events such as conferences or reviews",
    "product": "Named products or portfolios",
    "concept": "Abstract concepts that are not the types above",
    "date": "Dates or years",
}

# Extractor label -> stored entity_type. Product and concept share CONCEPT,
# matching the previous GLiNER2 backend.
LABEL_TO_TYPE: dict[str, str] = {
    "person": "PERSON",
    "organization": "ORG",
    "location": "PLACE",
    "event": "EVENT",
    "product": "CONCEPT",
    "concept": "CONCEPT",
    "date": "DATE",
}

# Stored / extractor type -> ontology label. Empty string means "unknown".
_TYPE_ALIASES: dict[str, str] = {
    "person": "person",
    "organization": "organization",
    "org": "organization",
    "location": "location",
    "place": "location",
    "event": "event",
    "product": "product",
    "concept": "concept",
    "date": "date",
    "misc": "",
}


def canonical_type(value: str | None) -> str:
    """Normalize PERSON / organization / place to an ontology label."""
    if not value:
        return ""
    return _TYPE_ALIASES.get(value.strip().lower().replace(" ", "_"), "")


def stored_relation(name: str) -> str:
    """``works for`` and ``works_for`` both become ``WORKS_FOR``."""
    return " ".join(name.strip().split()).upper().replace(" ", "_").replace("-", "_")


def display_relation(name: str) -> str:
    """Label text for a classifier. Parentheses are rejected by Decide."""
    return stored_relation(name).lower().replace("_", " ")


@dataclass(frozen=True)
class RelationSpec:
    """One typed relation. Head/tail are ontology labels, not stored types."""

    name: str
    head: tuple[str, ...]
    tail: tuple[str, ...]
    unique_head: bool = False
    symmetric: bool = False


# Typed endpoints for the closed relation set the pipeline already used,
# plus the constraints JointIE can enforce at decode time. ``works_for`` is
# not unique_head: a person can have a current employer and a previous one.
RELATION_SPECS: tuple[RelationSpec, ...] = (
    RelationSpec("works_for", ("person",), ("organization",)),
    RelationSpec("located_in", ("organization", "person", "event"), ("location",)),
    RelationSpec("founded", ("person",), ("organization",)),
    RelationSpec("leads", ("person",), ("organization",), unique_head=True),
    RelationSpec("partnered_with", ("organization",), ("organization",), symmetric=True),
    RelationSpec("acquired", ("organization",), ("organization",)),
    RelationSpec("invested_in", ("organization",), ("organization",)),
    RelationSpec("member_of", ("person",), ("organization",)),
    RelationSpec("subsidiary_of", ("organization",), ("organization",), unique_head=True),
)

# The extractor always finishes the JointIE schema with this constraint.
SCHEMA_STEPS: tuple[str, ...] = ("entities", "relations", "no_self_loops")


def relation_kwargs(spec: RelationSpec) -> dict:
    """Keyword arguments for ``JointSchema.relation``, minus the name."""
    head: str | list[str] = spec.head[0] if len(spec.head) == 1 else list(spec.head)
    tail: str | list[str] = spec.tail[0] if len(spec.tail) == 1 else list(spec.tail)
    kwargs: dict = {"head": head, "tail": tail}
    if spec.unique_head:
        kwargs["unique_head"] = True
    if spec.symmetric:
        kwargs["symmetric"] = True
    return kwargs


def legal_relations(head_type: str | None, tail_type: str | None) -> list[str]:
    """Stored relation names the ontology allows between these endpoint types."""
    head = canonical_type(head_type)
    tail = canonical_type(tail_type)
    if not head or not tail:
        return []
    names: list[str] = []
    for spec in RELATION_SPECS:
        forward = head in spec.head and tail in spec.tail
        backward = spec.symmetric and tail in spec.head and head in spec.tail
        if forward or backward:
            names.append(stored_relation(spec.name))
    return names


def legal_display_labels(head_type: str | None, tail_type: str | None) -> list[str]:
    """Plain-word labels for a Decide relation question, without ``no relation``."""
    return [display_relation(name) for name in legal_relations(head_type, tail_type)]
