"""Document profiles: what changes between kinds of documents, as settings.

One pipeline serves every document; a profile carries only what is specific
to a kind of document:

* the ontology — entity labels (with the descriptions GLiNER2.5 reads as
  part of the question) and the typed relation set;
* windows to skip at extraction (boilerplate such as an exhibit index);
* defaults for the gate profile and the excluded node types;
* default curation hides — generic nouns that are never actors in this kind
  of document;
* the gold fact file ``entity-graph eval`` scores against.

Select one with ``DOC_PROFILE`` (default ``general``). Explicit environment
settings (``EXCLUDE_ENTITY_TYPES``, ``SYSTEMONE_PROFILE``) still win over a
profile's defaults. See DECISIONS.md D26.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .extract.ontology import (
    ENTITY_DESCRIPTIONS,
    LABEL_TO_TYPE,
    RELATION_SPECS,
    Ontology,
    RelationSpec,
)


@dataclass(frozen=True)
class DocumentProfile:
    name: str
    description: str
    ontology: Ontology
    # A sentence window matching any of these is not extracted. Offsets are
    # untouched: the text is still stored and quoted, only not mined.
    skip_window_patterns: tuple[str, ...] = ()
    systemone_profile: str = ""
    exclude_entity_types: frozenset[str] = frozenset({"DATE"})
    default_hide: tuple[str, ...] = ()
    gold_path: str = ""

    def skips(self, window: str) -> bool:
        """True when this window is boilerplate the profile does not mine."""
        return any(p.search(window) for p in _compile(self.skip_window_patterns))


@lru_cache(maxsize=None)
def _compile(patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


GENERAL = DocumentProfile(
    name="general",
    description="Default ontology; no skipped windows. Use for news, reports, memos.",
    ontology=Ontology(dict(ENTITY_DESCRIPTIONS), dict(LABEL_TO_TYPE), RELATION_SPECS),
    gold_path="data/gold/greengrid_partnership.jsonl",
)


# --- SEC filings (S-1, 10-K, proxy statements) ------------------------------
# The S-1 evaluation: compute partnerships, litigation, underwriters, auditors
# and counsel had no relation type; "Claude" (378 mentions) was typed PERSON
# because nothing described products; the exhibit index and signature pages
# produced the worst edges (Anthropic ACQUIRED Google).
_SEC_ENTITIES = {
    "person": "Named individual people, not products, AI models, or groups",
    "organization": "Companies, funds, trusts, agencies, courts, universities, or banks",
    "location": "Cities, states, countries, or regions",
    "event": "Named events such as offerings, lawsuits, or acquisitions",
    "product": "Named products, AI models, software, platforms, or services",
    "concept": "Abstract concepts, securities, or contracts that are not the types above",
    "date": "Dates or years",
}
_SEC_LABELS = {**LABEL_TO_TYPE, "product": "PRODUCT"}
_SEC_RELATIONS = RELATION_SPECS + (
    RelationSpec("supplies", ("organization",), ("organization",)),
    RelationSpec("sued", ("organization", "person"), ("organization", "person")),
    RelationSpec("underwriter_of", ("organization",), ("organization",)),
    RelationSpec("auditor_of", ("organization",), ("organization",)),
    RelationSpec("advisor_to", ("organization", "person"), ("organization",)),
    RelationSpec("develops", ("organization",), ("product",)),
    # FIBO's lender/borrower distinction (D39): without it lenders and loan
    # agents had no relation and landed on supplies or advisor_to.
    RelationSpec("lends_to", ("organization",), ("organization",)),
)

SEC_FILING = DocumentProfile(
    name="sec_filing",
    description="SEC registration statements and reports: capital, governance, "
                "commercial and legal relations; exhibit index and signatures skipped.",
    ontology=Ontology(_SEC_ENTITIES, _SEC_LABELS, _SEC_RELATIONS),
    skip_window_patterns=(
        # Exhibit index rows, and link-dense windows (table of contents, image
        # galleries). One link is normal: every page repeats a "Table of
        # Contents" link, so a single-link rule would skip 514 prose windows.
        r"\]\(https?://www\.sec\.gov/Archives/edgar/data/[^)]*exhibit",
        r"(?s)(?:\]\(https?://www\.sec\.gov/Archives/edgar/data/[^)]*\).*?){3}",
        # Signature blocks.
        r"/s/\s*\S",
    ),
    # Generic role and group nouns of a filing. The registrant's own board is
    # NOT here: merge it into the registrant in the experiment's curation file,
    # or every director loses the edge that makes them a director.
    default_hide=(
        "directors", "director", "independent directors", "non-employee directors",
        "board members", "NEOs", "NEO", "officers", "officer", "executive officers",
        "executives", "financial officers", "employees", "employee", "co-founders",
        "founders", "agents", "attorneys-in-fact", "contractors", "consultants",
        "customers", "investors", "institutional investors", "individuals",
        "participants", "stockholders", "affiliates", "officials", "advisors",
        "recipients", "service providers", "vendors", "partners", "competitors",
        "controlling persons", "representatives", "senior leadership",
        "senior management team", "researchers", "companies", "corporation",
        "corporations", "prospectus", "public offerings", "tax returns", "VIE",
    ),
    gold_path="data/gold/anthropic_s1.jsonl",
)


# --- Histories and other long scholarly prose (OCR books) -------------------
_HISTORY_ENTITIES = {
    "person": "Named people",
    "organization": "Banks, firms, government bodies, political parties, "
                    "associations, armies, or newspapers",
    "location": "Cities, provinces, countries, or regions",
    "event": "Named events such as campaigns, conferences, strikes, or wars",
    "product": "Named products or commodities",
    "concept": "Abstract concepts, policies, or bonds that are not the types above",
    "date": "Dates or years",
}
_HISTORY_RELATIONS = RELATION_SPECS + (
    RelationSpec(
        "allied_with", ("person", "organization"), ("person", "organization"), symmetric=True
    ),
    RelationSpec("opposed", ("person", "organization"), ("person", "organization")),
)

HISTORY_BOOK = DocumentProfile(
    name="history_book",
    description="Scholarly histories (often OCR): alliances and opposition between "
                "people and institutions on top of the general relations.",
    ontology=Ontology(_HISTORY_ENTITIES, dict(LABEL_TO_TYPE), _HISTORY_RELATIONS),
    # Generic institution nouns a history uses for many different bodies.
    default_hide=(
        "bank", "banks", "central bank", "finance", "banking", "labor",
        "government agencies", "commission", "board", "board of directors",
        "board of supervisors", "corporation", "association", "officials",
        "council", "firm", "agencies", "treasury", "financial institutions",
        "chambers of commerce",
    ),
)

PROFILES: dict[str, DocumentProfile] = {
    profile.name: profile for profile in (GENERAL, SEC_FILING, HISTORY_BOOK)
}


def get_document_profile(name: str | None) -> DocumentProfile:
    """The named profile; empty selects ``general``. Unknown names raise."""
    key = (name or "general").strip().lower()
    try:
        return PROFILES[key]
    except KeyError:
        raise ValueError(
            f"Unknown document profile {name!r} (expected one of: {', '.join(PROFILES)})"
        ) from None


def all_relation_specs() -> tuple[RelationSpec, ...]:
    """Every relation any profile defines (names are unique across profiles)."""
    seen: dict[str, RelationSpec] = {}
    for profile in PROFILES.values():
        for spec in profile.ontology.relation_specs:
            seen.setdefault(spec.name, spec)
    return tuple(seen.values())
