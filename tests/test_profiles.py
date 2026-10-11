"""Prompt-profile tests — no network. Verifies profile selection and that
``judge_relations`` honors a profile's wording, criteria, examples, and batch
size, and maps provider modality labels onto the gate's labels.
"""

import pytest

from entity_graph.config import Config
from entity_graph.extract import jev, profiles


def _fake_client(monkeypatch, response):
    client = jev.SystemOneClient(Config(systemone_api_key="test-key"))
    calls: list = []

    def fake_decide(state, questions):
        calls.append((state, questions))
        return response

    monkeypatch.setattr(client, "decide", fake_decide)
    return client, calls


def test_get_profile_defaults_and_registry() -> None:
    assert profiles.get_profile("") is profiles.DEFAULT_PROFILE
    assert profiles.get_profile(None) is profiles.DEFAULT_PROFILE
    assert profiles.get_profile("clef") is profiles.CLEF_PROFILE
    with pytest.raises(ValueError):
        profiles.get_profile("nope")


def test_clef_profile_is_single_claim_with_examples() -> None:
    assert profiles.CLEF_PROFILE.batch_size == 1
    assert profiles.CLEF_PROFILE.examples


def test_judge_relations_uses_default_wording(monkeypatch) -> None:
    response = {"answers": {"claim_0": {"type": "noul", "noul": 0.8},
                            "modality_0": {"type": "choice", "choice": "fact"}}}
    client, calls = _fake_client(monkeypatch, response)
    verdicts = jev.judge_relations(client, [
        {"id": 0, "subject": "A", "relation": "WORKS_FOR", "object": "B",
         "evidence": "A works for B"},
    ])
    assert verdicts[0].assertion == 0.8
    assert verdicts[0].modality == "fact"
    state, questions = calls[0]
    assert "examples" not in state
    assert "criteria" not in questions["claim_0"]
    assert questions["modality_0"]["criteria"] is profiles.MODALITY_CRITERIA


def test_judge_relations_honors_profile_wording_and_batch(monkeypatch) -> None:
    client, calls = _fake_client(monkeypatch, {"answers": {}})
    claims = [
        {"id": i, "subject": "A", "relation": "FOUNDED", "object": "B",
         "evidence": "founded by B"}
        for i in range(3)
    ]
    jev.judge_relations(client, claims, profile=profiles.CLEF_PROFILE)
    assert len(calls) == 3  # one claim per request (profile batch size)
    state, questions = calls[0]
    assert state["examples"] == list(profiles.EXAMPLES)
    assert "is the doer" in questions["claim_0"]["instructions"]
    assert questions["modality_0"]["criteria"] is profiles.MODALITY_TENSE


def test_judge_relations_batch_size_override(monkeypatch) -> None:
    client, calls = _fake_client(monkeypatch, {"answers": {}})
    claims = [
        {"id": i, "subject": "A", "relation": "R", "object": "B", "evidence": "e"}
        for i in range(5)
    ]
    jev.judge_relations(client, claims, batch_size=2, profile=profiles.CLEF_PROFILE)
    assert len(calls) == 3  # 5 claims at 2 per request


def test_judge_relations_applies_modality_map(monkeypatch) -> None:
    response = {"answers": {"claim_0": {"type": "noul", "noul": 0.9},
                            "modality_0": {"type": "choice", "choice": "completed"}}}
    client, _ = _fake_client(monkeypatch, response)
    profile = profiles.RelationPromptProfile(
        name="x", task="t", assertion=profiles.assertion_plain,
        modality_criteria={"completed": "done", "planned": "later"},
        batch_size=4, modality_map={"completed": "fact", "planned": "forecast"},
    )
    verdicts = jev.judge_relations(client, [
        {"id": 0, "subject": "A", "relation": "R", "object": "B", "evidence": "e"},
    ], profile=profile)
    assert verdicts[0].modality == "fact"
