"""Shared ontology used by JointIE and the local Decide gate."""

from entity_graph.extract.ontology import (
    RELATION_SPECS,
    SCHEMA_STEPS,
    canonical_pair,
    is_symmetric,
    legal_relations,
    relation_kwargs,
    stored_relation,
)


def test_stored_relation_normalizes_display_names() -> None:
    assert stored_relation("works for") == "WORKS_FOR"
    assert stored_relation("WORKS_FOR") == "WORKS_FOR"
    assert stored_relation("partnered-with") == "PARTNERED_WITH"


def test_typed_endpoints_reject_illegal_pairs() -> None:
    assert "ACQUIRED" not in legal_relations("person", "organization")
    assert "ACQUIRED" in legal_relations("organization", "organization")
    assert "FOUNDED" in legal_relations("PERSON", "ORG")
    assert legal_relations("organization", "person") == []
    assert legal_relations("", "organization") == []


def test_schema_constraints_match_the_specs() -> None:
    by_name = {spec.name: spec for spec in RELATION_SPECS}
    assert relation_kwargs(by_name["subsidiary_of"])["unique_head"] is True
    assert relation_kwargs(by_name["leads"])["unique_head"] is True
    # The decoder drops or fails on symmetric relations; they are
    # canonicalized after decoding instead (D21).
    assert "symmetric" not in relation_kwargs(by_name["partnered_with"])
    assert by_name["partnered_with"].symmetric is True
    assert "unique_head" not in relation_kwargs(by_name["founded"])
    assert relation_kwargs(by_name["located_in"])["head"] == [
        "organization", "person", "event",
    ]
    assert "no_self_loops" in SCHEMA_STEPS
    assert all(not getattr(spec, "allow_self", False) for spec in RELATION_SPECS)


def test_symmetric_relations_have_one_canonical_order() -> None:
    assert is_symmetric("PARTNERED_WITH")
    assert is_symmetric("partnered with")
    assert not is_symmetric("WORKS_FOR")
    assert canonical_pair("PARTNERED_WITH", "Google", "Anthropic") is True
    assert canonical_pair("PARTNERED_WITH", "Anthropic", "Google") is False
    assert canonical_pair("ACQUIRED", "Google", "Anthropic") is False


def test_registry_defines_each_relation_once_with_a_known_module_and_group() -> None:
    from entity_graph.extract.ontology import (
        RELATION_GROUPS,
        RELATION_REGISTRY,
        SCHEMA_MODULES,
    )

    names = [spec.name for spec in RELATION_REGISTRY]
    assert len(names) == len(set(names))
    for spec in RELATION_REGISTRY:
        assert spec.module in SCHEMA_MODULES
        assert spec.group in RELATION_GROUPS
        assert all(standard and term for standard, term in spec.maps)


def test_relations_for_keeps_registry_order_and_rejects_unknown_modules() -> None:
    import pytest

    from entity_graph.extract.ontology import RELATION_REGISTRY, relations_for

    chosen = relations_for(("finance", "core"))
    assert [s for s in RELATION_REGISTRY if s in chosen] == list(chosen)
    assert {s.module for s in chosen} == {"core", "finance"}
    assert relations_for(("core",)) == RELATION_SPECS
    with pytest.raises(ValueError):
        relations_for(("core", "astrology"))


def test_standard_terms_name_a_known_standard_and_wikidata_ids_are_properties() -> None:
    import re

    from entity_graph.extract.ontology import RELATION_REGISTRY

    known = {"ACE2005", "TAC-KBP", "Wikidata", "schema.org", "FIBO"}
    for spec in RELATION_REGISTRY:
        for standard, term in spec.maps:
            assert standard in known
            if standard == "Wikidata":
                assert re.fullmatch(r"\^?P\d+", term), (spec.name, term)


def test_cues_match_whole_words_and_word_starts_in_any_case() -> None:
    from entity_graph.extract.ontology import RelationSpec, cued

    spec = RelationSpec("parent_of", ("person",), ("person",), cues=("son", "marri*", "born to"))
    assert cued(spec, "He left his SON the estate.")
    assert cued(spec, "They were married in 1901.")
    assert cued(spec, "She was born to a farming family.")
    assert not cued(spec, "Mason sold the season's crop.")  # "son" is a whole word
    assert not cued(spec, "His sons inherited it.")
    assert cued(RelationSpec("x", ("person",), ("person",)), "any text at all")  # no cues


def test_specialist_relations_have_cues_and_the_core_basics_are_always_listed() -> None:
    from entity_graph.extract.ontology import RELATION_REGISTRY, listed_relations

    for spec in RELATION_REGISTRY:
        assert all(cue and cue == cue.strip().lower() for cue in spec.cues), spec.name
        if spec.module != "core":
            assert spec.cues, spec.name
    listed = {s.name for s in listed_relations(RELATION_REGISTRY, "Dana Wu works at Acme.")}
    assert {"works_for", "located_in", "member_of", "subsidiary_of"} <= listed
    assert not listed & {"underwriter_of", "sued", "spouse_of", "educated_at"}
    listed = {s.name for s in listed_relations(RELATION_REGISTRY,
                                               "Goldman Sachs is an underwriter of the offering.")}
    assert "underwriter_of" in listed
