"""Shared entity and relation schema for extraction and the local gate.

JointIE and GLiNER2.5-Decide must agree on which relations are legal between
which types. One definition here; both call sites read it. See DECISIONS.md D13.
"""

from __future__ import annotations

import functools
import re
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

    ``cues`` are words a text uses when it states the relation (whole words or
    phrases, any case; a trailing ``*`` matches a word start). With
    ``RELATION_CUES`` on, an LLM prompt lists a relation with cues only for a
    window holding one (D47); a relation without cues is always listed.
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
    cues: tuple[str, ...] = ()


# Parents in the relation hierarchy. A relation's group is what a reader can
# still be credited with when it picks a sibling, and what a document type
# with no module of its own falls back on.
RELATION_GROUPS: dict[str, str] = {
    "affiliation": "a person's tie to an organization: employment, office, membership, "
                   "founding (ACE ORG-AFF)",
    "ownership": "control or equity between organizations: parent, acquirer, shareholder",
    "commerce": "goods, services or money provided by one organization to another",
    "cooperation": "a stated partnership or alliance, either way round",
    "family": "kinship or marriage between people (ACE PER-SOC.Family)",
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

# Cue words are written from each relation's meaning, not from any gold text.
# Kinship and marriage words are shared by the family relations: a passage
# naming one kin tie often states another ("her brother's wife").
_KIN_CUES = (
    "wife", "wives", "husband*", "spouse*", "marri*", "marry", "wed", "wedded", "widow*",
    "divorc*", "consort", "father*", "mother*", "parent*", "son", "sons", "daughter*", "child",
    "children", "born to", "brother*", "sister*", "sibling*", "twin*", "née", "heir*", "grand*",
    "step*", "family", "families", "dad", "mom", "mum", "uncle*", "aunt*", "nephew*", "niece*",
    "cousin*", "in-law*",
)

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
                "suppliers",
        cues=("suppl*", "customer*", "client*", "purchas*", "buy*", "bought", "sell*", "sold",
              "sale*", "vendor*", "provid*", "procur*", "contract*", "agreement*", "licens*",
              "licenc*", "distribut*", "resell*", "order*", "service*", "source*", "sourcing",
              "deliver*", "manufactur*", "partner*", "capacity", "facilit*")),
    RelationSpec(
        "sued", ("organization", "person"), ("organization", "person"), module="legal",
        group="conflict",
        meaning="head filed a lawsuit or claim against tail",
        maps=((_ACE, "event:Justice.Sue"),),
        cues=("sue", "sued", "sues", "suing", "lawsuit*", "suit", "suits", "litigat*",
              "complaint*", "court*", "plaintiff*", "defendant*", "alleg*", "claim*",
              "arbitrat*", "settle*", "judgment*", "judgement*", "proceeding*", "infring*",
              "petition*")),
    RelationSpec(
        "underwriter_of", ("organization",), ("organization",), module="finance",
        group="commerce",
        meaning="head is an underwriter of tail's securities offering; arrangers and "
                "bookrunners of a loan are not underwriters",
        cues=("underwrit*", "bookrunn*", "book-runn*", "book runn*", "lead manager*",
              "placement agent*", "initial purchaser*", "dealer*", "offering*")),
    RelationSpec(
        "auditor_of", ("organization",), ("organization",), module="finance", group="commerce",
        meaning="head is the independent auditor (registered public accounting firm) of tail",
        maps=((_WD, "^P8571"),),
        cues=("audit*", "accountant*", "accounting firm*", "public accounting")),
    RelationSpec(
        "advisor_to", ("organization", "person"), ("organization",), module="commerce",
        group="commerce",
        meaning="head advises tail, including as its legal counsel; lenders and agents of a "
                "loan are not advisors",
        cues=("advis*", "counsel*", "consult*", "legal", "passed upon", "law firm*", "attorney*",
              "lawyer*", "banker*", "represent*")),
    RelationSpec(
        "develops", ("organization",), ("product",), module="commerce", group="creation",
        meaning="head develops or makes tail",
        maps=((_ACE, "ART.User-Owner-Inventor-Manufacturer"), (_WD, "^P178"),
              (_WD, "^P176"), (_SO, "^manufacturer")),
        cues=("develop*", "built", "build*", "design*", "creat*", "launch*", "introduc*",
              "releas*", "product*", "model*", "platform*", "engineer*", "invent*",
              "manufactur*", "make*", "made", "maker*", "produc*", "brand*", "software",
              "game*", "publish*")),
    # FIBO's lender/borrower distinction (D39): without it lenders and loan
    # agents had no relation and landed on supplies or advisor_to.
    RelationSpec(
        "lends_to", ("organization",), ("organization",), module="finance", group="commerce",
        meaning="head lends money to tail, or is a lender, administrative agent, collateral "
                "agent or arranger of a loan, credit facility or notes extended to tail",
        maps=((_FIBO, "FBC/DebtAndEquities/Debt: lender -> Borrower"),),
        cues=("loan*", "lend*", "credit*", "borrow*", "debt*", "facilit*", "note", "notes", "bond*",
              "financ*", "mortgage*", "indebted*", "revolv*")),
    RelationSpec(
        "allied_with", ("person", "organization"), ("person", "organization"), symmetric=True,
        module="politics", group="cooperation",
        cues=("ally", "allied", "allies", "alliance*", "coalition*", "treat*", "pact*", "support*",
              "join*", "unite*", "united", "cooperat*", "collaborat*", "side*")),
    RelationSpec(
        "opposed", ("person", "organization"), ("person", "organization"), module="politics",
        group="conflict",
        cues=("oppos*", "against", "enem*", "rival*", "war", "wars", "fought", "fight*", "battle*",
              "conflict*", "resist*", "rebel*", "attack*", "invad*", "invasion*", "struggl*",
              "hostil*", "denounc*", "protest*", "boycott*", "defeat*", "overthr*")),
    # Core relations the standards share and the registry lacked (D41, D43):
    # family and education, the largest gaps between people and organizations
    # on Re-DocRED (D42).
    RelationSpec(
        "spouse_of", ("person",), ("person",), symmetric=True, group="family",
        meaning="head and tail are or were married to each other",
        maps=((_ACE, "PER-SOC.Family"), (_KBP, "per:spouse"), (_WD, "P26"), (_SO, "spouse")),
        cues=_KIN_CUES),
    RelationSpec(
        "parent_of", ("person",), ("person",), group="family",
        meaning="head is the mother or father of tail",
        maps=((_ACE, "PER-SOC.Family"), (_KBP, "per:children"), (_WD, "P40"), (_WD, "^P22"),
              (_WD, "^P25"), (_SO, "children")),
        cues=_KIN_CUES),
    RelationSpec(
        "sibling_of", ("person",), ("person",), symmetric=True, group="family",
        meaning="head and tail are brothers or sisters",
        maps=((_ACE, "PER-SOC.Family"), (_KBP, "per:siblings"), (_WD, "P3373"),
              (_SO, "sibling")),
        cues=_KIN_CUES),
    RelationSpec(
        "educated_at", ("person",), ("organization",), group="affiliation",
        meaning="head studied at, attended or graduated from tail, a school, college or "
                "university; teaching or research there is works_for",
        maps=((_ACE, "ORG-AFF.Student-Alum"), (_KBP, "per:schools_attended"), (_WD, "P69"),
              (_SO, "alumniOf")),
        cues=("universit*", "college*", "school*", "institut*", "academy", "academi*", "degree*",
              "graduat*", "alumn*", "studied", "study*", "studies", "student*", "educat*",
              "enrol*", "attended", "b.a.", "b.s.", "m.a.", "m.s.", "mba", "m.b.a.", "ph.d*",
              "phd", "j.d.", "m.d.", "ll.b*", "ll.m*", "bachelor*", "master*", "doctor*",
              "polytechnic*", "conservator*", "seminar*", "faculty", "professor*",
              "lecturer*", "fellow*", "scholar*", "thesis", "dissertation*", "taught",
              "teach*", "tutor*")),
)


def relations_for(modules: tuple[str, ...]) -> tuple[RelationSpec, ...]:
    """The registry's relations in ``modules``, in registry order. Unknown modules raise."""
    unknown = set(modules) - set(SCHEMA_MODULES)
    if unknown:
        raise ValueError(f"Unknown schema module(s): {', '.join(sorted(unknown))}")
    return tuple(spec for spec in RELATION_REGISTRY if spec.module in modules)


# The general ontology's relations: the core module.
RELATION_SPECS: tuple[RelationSpec, ...] = relations_for(("core",))


@functools.lru_cache(maxsize=None)
def _cue_pattern(cues: tuple[str, ...]) -> re.Pattern:
    parts = (rf"(?<!\w){re.escape(cue.rstrip('*'))}" + ("" if cue.endswith("*") else r"(?!\w)")
             for cue in cues)
    return re.compile("|".join(parts), re.IGNORECASE)


def cued(spec: RelationSpec, text: str) -> bool:
    """Whether ``text`` holds one of the relation's cues (always true without cues)."""
    return not spec.cues or bool(_cue_pattern(spec.cues).search(text))


def listed_relations(specs, text: str) -> tuple[RelationSpec, ...]:
    """The relations of ``specs`` an LLM prompt lists for the window ``text`` (D47)."""
    return tuple(spec for spec in specs if cued(spec, text))

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
