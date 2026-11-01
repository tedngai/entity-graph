"""Document profiles — pure, no DB or model. See DECISIONS.md D26."""

from __future__ import annotations

import pytest

from entity_graph.config import load_config
from entity_graph.doc_profiles import (
    PROFILES,
    all_relation_specs,
    get_document_profile,
)
from entity_graph.extract.gliner_extractor import parse_joint_payload
from entity_graph.extract.ontology import RELATION_SPECS, is_symmetric, legal_relations


def test_registry_and_lookup() -> None:
    assert set(PROFILES) == {"general", "sec_filing", "history_book"}
    assert get_document_profile("").name == "general"
    assert get_document_profile("SEC_FILING").name == "sec_filing"
    with pytest.raises(ValueError):
        get_document_profile("novel")


def test_general_profile_is_the_existing_ontology() -> None:
    assert get_document_profile("general").ontology.relation_specs == RELATION_SPECS


def test_profiles_extend_rather_than_replace_the_general_relations() -> None:
    general = {spec.name for spec in RELATION_SPECS}
    for profile in PROFILES.values():
        assert general <= {spec.name for spec in profile.ontology.relation_specs}


def test_sec_profile_types_products_and_adds_filing_relations() -> None:
    sec = get_document_profile("sec_filing").ontology
    assert sec.label_to_type["product"] == "PRODUCT"
    names = {spec.name for spec in sec.relation_specs}
    assert {"supplies", "sued", "underwriter_of", "auditor_of", "advisor_to", "develops"} <= names
    assert "SUED" in legal_relations("organization", "organization", sec.relation_specs)
    assert "SUED" not in legal_relations("organization", "organization")
    assert legal_relations("organization", "product", sec.relation_specs) == ["DEVELOPS"]


def test_symmetry_is_known_across_profiles() -> None:
    assert is_symmetric("ALLIED_WITH")
    assert not is_symmetric("OPPOSED")
    assert len({spec.name for spec in all_relation_specs()}) == len(all_relation_specs())


def test_sec_profile_skips_exhibit_index_and_signature_windows() -> None:
    sec = get_document_profile("sec_filing")
    exhibit = ("| 3.1 | | [Amended and Restated Certificate of Incorporation]"
               "(https://www.sec.gov/Archives/edgar/data/2130357/000162828026067168/"
               "exhibit31-sx1.htm) |")
    signature = "| By: | /s/ Dario Amodei | | Chief Executive Officer |"
    prose = "In April 2026, the Company completed the acquisition of Coefficient Bio, Inc."
    assert sec.skips(exhibit)
    assert sec.skips(signature)
    assert not sec.skips(prose)
    assert not get_document_profile("general").skips(exhibit)


def test_profile_label_types_reach_parsed_mentions() -> None:
    text = "Anthropic develops Claude."
    payload = {"entities": [
        {"id": "a", "type": "organization", "text": "Anthropic", "start": 0, "end": 9},
        {"id": "b", "type": "product", "text": "Claude", "start": 19, "end": 25},
    ], "relations": []}
    sec = get_document_profile("sec_filing").ontology
    types = {m.surface: m.entity_type
             for m in parse_joint_payload(payload, text, label_to_type=sec.label_to_type).mentions}
    assert types == {"Anthropic": "ORG", "Claude": "PRODUCT"}
    general = {m.surface: m.entity_type for m in parse_joint_payload(payload, text).mentions}
    assert general["Claude"] == "CONCEPT"


def test_profile_defaults_yield_to_explicit_environment(monkeypatch) -> None:
    monkeypatch.delenv("EXCLUDE_ENTITY_TYPES", raising=False)
    monkeypatch.delenv("SYSTEMONE_PROFILE", raising=False)
    monkeypatch.setenv("DOC_PROFILE", "sec_filing")
    config = load_config()
    assert config.doc_profile == "sec_filing"
    assert config.exclude_entity_types == frozenset({"DATE"})
    monkeypatch.setenv("EXCLUDE_ENTITY_TYPES", "")
    monkeypatch.setenv("SYSTEMONE_PROFILE", "clef")
    config = load_config()
    assert config.exclude_entity_types == frozenset()
    assert config.systemone_profile == "clef"


def test_sec_default_hides_keep_the_board() -> None:
    hides = {name.lower() for name in get_document_profile("sec_filing").default_hide}
    assert "directors" in hides
    assert "board of directors" not in hides
