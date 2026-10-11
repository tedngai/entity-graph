"""Jev client contract tests — no network. Uses the documented response shape
(https://docs.typesafe.ai/api) with a stubbed ``decide``.
"""

from entity_graph.config import Config
from entity_graph.extract import jev


def _client(monkeypatch, response):
    client = jev.SystemOneClient(Config(systemone_api_key="test-key"))
    captured: dict = {}

    def fake_decide(state, questions):
        captured["state"] = state
        captured["questions"] = questions
        return response

    monkeypatch.setattr(client, "decide", fake_decide)
    return client, captured


def test_align_pair_parses_documented_response(monkeypatch) -> None:
    response = {
        "model": "jev-1.13.0",
        "answers": {
            "same_entity": {"type": "score", "score": 1.9, "confidence": 0.9},
            "name_match": {"type": "noul", "noul": 0.95},
            "kind_match": {"type": "noul", "noul": 0.80},
            "context_match": {"type": "noul", "noul": 0.70},
        },
    }
    client, captured = _client(monkeypatch, response)
    verdict = jev.align_pair(client, {"name": "A"}, {"name": "B"})

    assert verdict.outcome == "merge"
    assert verdict.score == 1.9
    assert verdict.sidecars == {"name": 0.95, "kind": 0.80, "context": 0.70}

    questions = captured["questions"]
    assert questions["same_entity"]["type"] == "score"
    assert questions["same_entity"]["instructions"]
    assert questions["same_entity"]["criteria"] is jev.ALIGNMENT_CRITERIA
    assert questions["name_match"]["type"] == "noul"
    assert questions["name_match"]["instructions"]


def test_align_pair_routes_by_rounding(monkeypatch) -> None:
    for score, expected in [(0.4, "separate"), (1.1, "curator"), (2.0, "merge")]:
        response = {"answers": {"same_entity": {"type": "score", "score": score}}}
        client, _ = _client(monkeypatch, response)
        verdict = jev.align_pair(client, {"name": "A"}, {"name": "B"})
        assert verdict.outcome == expected, (score, verdict.outcome)


def test_validates_relation_parses_noul(monkeypatch) -> None:
    response = {"answers": {"claim_0": {"type": "noul", "noul": 0.91}}}
    client, captured = _client(monkeypatch, response)
    probability = jev.validates_relation(client, "A works for B", "A", "WORKS_FOR", "B")
    assert probability == 0.91
    assert captured["questions"]["claim_0"]["type"] == "noul"
    assert captured["questions"]["claim_0"]["instructions"]


def test_validate_relations_batches_and_parses(monkeypatch) -> None:
    response = {
        "answers": {
            "claim_1": {"type": "noul", "noul": 0.9},
            "claim_2": {"type": "noul", "noul": 0.1},
        }
    }
    client, captured = _client(monkeypatch, response)
    verdicts = jev.validate_relations(
        client,
        [
            {"id": 1, "subject": "A", "relation": "WORKS_FOR", "object": "B",
             "evidence": "A works for B."},
            {"id": 2, "subject": "C", "relation": "WORKS_FOR", "object": "D",
             "evidence": "C is unrelated to D."},
        ],
    )
    assert verdicts == {1: 0.9, 2: 0.1}
    assert set(captured["questions"]) == {"claim_1", "claim_2"}
    assert captured["state"]["claims"][0]["id"] == 1


def test_numeric_answer_handles_missing_and_bare_values() -> None:
    assert jev._numeric_answer({"answers": {"q": {"type": "noul", "noul": 0.5}}}, "q", "noul") == 0.5
    assert jev._numeric_answer({"answers": {"q": 0.7}}, "q", "noul") == 0.7
    assert jev._numeric_answer({"answers": {}}, "q", "noul") is None
    assert jev._numeric_answer({"answers": {"q": {"type": "noul"}}}, "q", "noul") is None
