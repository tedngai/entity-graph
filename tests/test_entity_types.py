"""Majority-vote entity typing — pure, no DB. See DECISIONS.md D27."""

from __future__ import annotations

from entity_graph.doc_profiles import get_document_profile
from entity_graph.entity_types import entity_types_allow, vote_type

SEC = get_document_profile("sec_filing").ontology.relation_specs


def test_plurality_beats_the_first_mention() -> None:
    # The S-1 counts: Claude was stored PERSON, Anthropic PRODUCT.
    assert vote_type({"PRODUCT": 214, "ORG": 123, "PERSON": 109, "CONCEPT": 2}, "PERSON") == "PRODUCT"
    assert vote_type({"ORG": 105, "PRODUCT": 5, "CONCEPT": 2, "PLACE": 2}, "PRODUCT") == "ORG"


def test_ties_keep_the_current_type_else_break_alphabetically() -> None:
    assert vote_type({"ORG": 2, "PERSON": 2}, "PERSON") == "PERSON"
    assert vote_type({"ORG": 2, "PERSON": 2}, "PLACE") == "ORG"
    assert vote_type({"ORG": 2, "PERSON": 2}) == "ORG"


def test_untyped_mentions_vote_only_when_nothing_else_does() -> None:
    assert vote_type({"MISC": 9, None: 3, "ORG": 1}, None) == "ORG"
    assert vote_type({"MISC": 2}, None) == "MISC"
    assert vote_type({}, "ORG") == "ORG"


def test_entity_types_gate_relations() -> None:
    assert not entity_types_allow("WORKS_FOR", "PRODUCT", "ORG", SEC)
    assert entity_types_allow("WORKS_FOR", "PERSON", "ORG", SEC)
    assert entity_types_allow("DEVELOPS", "ORG", "PRODUCT", SEC)
    # Symmetric relations are legal in either order.
    assert entity_types_allow("PARTNERED_WITH", "ORG", "ORG", SEC)
    # Unknown or MISC types never block: there is nothing to arbitrate.
    assert entity_types_allow("WORKS_FOR", None, "ORG", SEC)
    assert entity_types_allow("WORKS_FOR", "MISC", "ORG", SEC)
