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
    """One typed relation. Head/tail are ontology labels, not stored types.

    ``meaning`` is the head -> tail definition an LLM reads (GLiNER ignores
    it). ``module`` is the schema module that carries the relation, ``group``
    its parent in the relation hierarchy (``RELATION_GROUPS``), and ``maps``
    its terms in outside standards as (standard, term) pairs; a term starting
    with ``^`` is the inverse property (tail -> head). See DECISIONS.md D40 and
    docs/SCHEMA_MAPPING.md (D41).
    """

    name: str
    head: tuple[str, ...]
    tail: tuple[str, ...]
    unique_head: bool = False
    symmetric: bool = False
    meaning: str = ""
    module: str = "core"
    group: str = ""
    maps: tuple[tuple[str, str], ...] = ()


# Parents in the relation hierarchy. A relation's group is what a reader can
# still be credited with when it picks a sibling, and what a document type
# with no module of its own falls back on.
RELATION_GROUPS: dict[str, str] = {
    "affiliation": "a person's tie to an organization: employment, office, membership, "
                   "founding (ACE ORG-AFF)",
    "ownership": "control or equity between organizations: parent, acquirer, shareholder",
    "commerce": "goods, services or money provided by one organization to another",
    "cooperation": "a stated partnership or alliance, either way round",
    "conflict": "a lawsuit or stated opposition",
    "location": "where an entity is",
    "creation": "who makes or develops a product",
}

# Schema modules. A document profile is a choice of modules; ``core`` holds
# the relations any text can state. The order of RELATION_REGISTRY is the
# order relations are listed to a model, so new relations go at the end.
SCHEMA_MODULES: dict[str, str] = {
    "core": "relations between people, organizations and places that any text can state",
    "commerce": "suppliers, customers, advisors and products",
    "finance": "lending, underwriting and auditing (FIBO's financial roles)",
    "legal": "litigation",
    "politics": "alliances and opposition between people and institutions",
}

_ACE, _KBP, _WD, _SO, _FIBO = "ACE2005", "TAC-KBP", "Wikidata", "schema.org", "FIBO"

# Every relation any profile can use, defined once. ``works_for`` is not
# unique_head: a person can have a current employer and a previous one.
RELATION_REGISTRY: tuple[RelationSpec, ...] = (
    RelationSpec(
        "works_for", ("person",), ("organization",), group="affiliation",
        meaning="head is or was employed by, or holds or held an officer or staff role at, "
                "tail; a board seat alone is member_of, not works_for",
        maps=((_ACE, "ORG-AFF.Employment"), (_KBP, "per:employee_or_member_of"),
              (_WD, "P108"), (_SO, "worksFor"))),
    RelationSpec(
        "located_in", ("organization", "person", "event"), ("location",), group="location",
        meaning="head is located, headquartered or has an office in tail; the state it is "
                "incorporated in does not count, and an address counts only for its owner",
        maps=((_ACE, "PHYS.Located"), (_ACE, "GEN-AFF.Org-Location"),
              (_KBP, "org:city_of_headquarters"), (_WD, "P276"), (_WD, "P159"),
              (_WD, "P131"), (_SO, "location"))),
    RelationSpec(
        "founded", ("person",), ("organization",), group="affiliation",
        meaning="head founded or co-founded tail",
        maps=((_ACE, "ORG-AFF.Founder"), (_KBP, "^org:founded_by"), (_WD, "^P112"),
              (_SO, "^founder"))),
    RelationSpec(
        "leads", ("person",), ("organization",), unique_head=True, group="affiliation",
        meaning="head is the chief executive, president, chair, manager, managing member or "
                "head of tail (including chair of a board committee); other officers "
                "(finance, operations, science) are works_for, and heading a team inside tail "
                "is not leading tail",
        maps=((_ACE, "ORG-AFF.Employment"), (_KBP, "^org:top_members_employees"),
              (_WD, "^P169"), (_WD, "^P488"), (_WD, "^P1037"))),
    RelationSpec(
        "partnered_with", ("organization",), ("organization",), symmetric=True,
        group="cooperation",
        meaning="head and tail have a stated partnership or collaboration; a provider or "
                "customer with no stated partnership is supplies",
        maps=((_WD, "P2652"),)),
    RelationSpec(
        "acquired", ("organization",), ("organization",), group="ownership",
        meaning="head acquired or agreed to acquire tail",
        maps=((_ACE, "event:Business.Merge-Org"),)),
    RelationSpec(
        "invested_in", ("organization",), ("organization",), group="ownership",
        meaning="head invested in, or holds equity bought in, tail, including a named "
                "beneficial owner of tail's stock",
        maps=((_ACE, "ORG-AFF.Investor-Shareholder"), (_KBP, "^org:shareholders"))),
    RelationSpec(
        "member_of", ("person",), ("organization",), group="affiliation",
        meaning="head is a member, director or trustee of tail (a board, committee, company, "
                "trust or group)",
        maps=((_ACE, "ORG-AFF.Membership"), (_KBP, "per:employee_or_member_of"),
              (_WD, "P463"), (_WD, "^P3320"), (_SO, "memberOf"))),
    RelationSpec(
        "subsidiary_of", ("organization",), ("organization",), unique_head=True,
        group="ownership",
        meaning="head is a subsidiary of, or wholly owned by, tail; a trust or fund that only "
                "holds shares or voting rights is not a subsidiary",
        maps=((_ACE, "PART-WHOLE.Subsidiary"), (_KBP, "org:parents"), (_WD, "P749"),
              (_WD, "^P355"), (_SO, "parentOrganization"))),
    RelationSpec(
        "supplies", ("organization",), ("organization",), module="commerce", group="commerce",
        meaning="head sells or provides goods, services, capacity or facilities to tail, "
                "including a customer named as such; lenders and agents of a loan are not "
                "suppliers"),
    RelationSpec(
        "sued", ("organization", "person"), ("organization", "person"), module="legal",
        group="conflict",
        meaning="head filed a lawsuit or claim against tail",
        maps=((_ACE, "event:Justice.Sue"),)),
    RelationSpec(
        "underwriter_of", ("organization",), ("organization",), module="finance",
        group="commerce",
        meaning="head is an underwriter of tail's securities offering; arrangers and "
                "bookrunners of a loan are not underwriters"),
    RelationSpec(
        "auditor_of", ("organization",), ("organization",), module="finance", group="commerce",
        meaning="head is the independent auditor (registered public accounting firm) of tail",
        maps=((_WD, "^P8571"),)),
    RelationSpec(
        "advisor_to", ("organization", "person"), ("organization",), module="commerce",
        group="commerce",
        meaning="head advises tail, including as its legal counsel; lenders and agents of a "
                "loan are not advisors"),
    RelationSpec(
        "develops", ("organization",), ("product",), module="commerce", group="creation",
        meaning="head develops or makes tail",
        maps=((_ACE, "ART.User-Owner-Inventor-Manufacturer"), (_WD, "^P178"),
              (_WD, "^P176"), (_SO, "^manufacturer"))),
    # FIBO's lender/borrower distinction (D39): without it lenders and loan
    # agents had no relation and landed on supplies or advisor_to.
    RelationSpec(
        "lends_to", ("organization",), ("organization",), module="finance", group="commerce",
        meaning="head lends money to tail, or is a lender, administrative agent, collateral "
                "agent or arranger of a loan, credit facility or notes extended to tail",
        maps=((_FIBO, "FBC/DebtAndEquities/Debt: lender -> Borrower"),)),
    RelationSpec(
        "allied_with", ("person", "organization"), ("person", "organization"), symmetric=True,
        module="politics", group="cooperation"),
    RelationSpec(
        "opposed", ("person", "organization"), ("person", "organization"), module="politics",
        group="conflict"),
)


def relations_for(modules: tuple[str, ...]) -> tuple[RelationSpec, ...]:
    """The registry's relations in ``modules``, in registry order. Unknown modules raise."""
    unknown = set(modules) - set(SCHEMA_MODULES)
    if unknown:
        raise ValueError(f"Unknown schema module(s): {', '.join(sorted(unknown))}")
    return tuple(spec for spec in RELATION_REGISTRY if spec.module in modules)


# The general ontology's relations: the core module.
RELATION_SPECS: tuple[RelationSpec, ...] = relations_for(("core",))

@dataclass(frozen=True)
class Ontology:
    """Entity labels (with descriptions), their stored types, and relations.

    The module constants below are the general ontology; document profiles
    (``doc_profiles.py``, D26) define others.
    """

    entity_descriptions: dict[str, str]
    label_to_type: dict[str, str]
    relation_specs: tuple[RelationSpec, ...]


# The extractor always finishes the JointIE schema with this constraint.
SCHEMA_STEPS: tuple[str, ...] = ("entities", "relations", "no_self_loops")


# The phrase JointIE reads for a relation, when it differs from the stored
# name. Only the label reaches the model (relation descriptions are accepted
# but ignored: identical output with and without them), and the name matters:
# "supplies" was never emitted, even for "Anthropic purchases compute capacity
# from Fluidstack", while "sells to" finds the supplier with the same head ->
# tail direction. Keyed by relation name so every profile shares one wording.
RELATION_LABELS: dict[str, str] = {
    "supplies": "sells to",
}


def relation_label(name: str) -> str:
    """The label JointIE is asked for this stored relation name."""
    return RELATION_LABELS.get(name, name)


def relation_kwargs(spec: RelationSpec) -> dict:
    """Keyword arguments for ``JointSchema.relation``, minus the name.

    ``symmetric`` is deliberately not passed to the decoder: with it, JointIE
    either never emits the relation or returns an empty, infeasible window
    (0 PARTNERED_WITH proposals across the S-1 and the Shanghai book). The
    relation is decoded directed and canonicalized afterwards by
    ``canonical_pair``. See DECISIONS.md D21.
    """
    head: str | list[str] = spec.head[0] if len(spec.head) == 1 else list(spec.head)
    tail: str | list[str] = spec.tail[0] if len(spec.tail) == 1 else list(spec.tail)
    kwargs: dict = {"head": head, "tail": tail}
    if spec.unique_head:
        kwargs["unique_head"] = True
    return kwargs


def is_symmetric(relation: str) -> bool:
    """True when the stored relation name is a symmetric spec (A~B == B~A).

    Checked against every profile's relations, so a symmetric relation is
    stored undirected whichever profile proposed it.
    """
    from ..doc_profiles import all_relation_specs

    name = stored_relation(relation)
    return any(
        spec.symmetric and stored_relation(spec.name) == name for spec in all_relation_specs()
    )


def canonical_pair(relation: str, source: str, target: str) -> bool:
    """True when (source, target) must be swapped to the canonical order.

    Symmetric relations are stored once, with endpoints in case-insensitive
    lexical order, so ``A partnered_with B`` and ``B partnered_with A`` collapse.
    """
    return is_symmetric(relation) and source.casefold() > target.casefold()


def legal_relations(
    head_type: str | None,
    tail_type: str | None,
    specs: tuple[RelationSpec, ...] | None = None,
) -> list[str]:
    """Stored relation names the ontology allows between these endpoint types."""
    head = canonical_type(head_type)
    tail = canonical_type(tail_type)
    if not head or not tail:
        return []
    names: list[str] = []
    for spec in RELATION_SPECS if specs is None else specs:
        forward = head in spec.head and tail in spec.tail
        backward = spec.symmetric and tail in spec.head and head in spec.tail
        if forward or backward:
            names.append(stored_relation(spec.name))
    return names


def legal_display_labels(
    head_type: str | None,
    tail_type: str | None,
    specs: tuple[RelationSpec, ...] | None = None,
) -> list[str]:
    """Plain-word labels for a Decide relation question, without ``no relation``."""
    return [display_relation(name) for name in legal_relations(head_type, tail_type, specs)]
