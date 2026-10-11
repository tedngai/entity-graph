"""Promotion rule, Jev modality question, and the injectable Decide gate."""

from entity_graph.extract.decide import judge_relations as decide_judge
from entity_graph.extract.gate import RelationJudgment, should_promote
from entity_graph.extract.jev import (
    RELATION_VALIDATION_THRESHOLD,
    SystemOneClient,
    judge_relations,
)
from entity_graph.config import Config


THRESHOLD = RELATION_VALIDATION_THRESHOLD


def test_cutoff_stays_frozen() -> None:
    assert RELATION_VALIDATION_THRESHOLD == 0.5


def test_unscored_proposals_promote() -> None:
    assert should_promote(None, threshold=THRESHOLD)
    assert should_promote(RelationJudgment(), threshold=THRESHOLD)


def test_jev_rejects_low_noul_and_non_facts() -> None:
    assert should_promote(
        RelationJudgment(assertion=0.6, modality="fact", source="jev"),
        threshold=THRESHOLD,
    )
    assert not should_promote(
        RelationJudgment(assertion=0.4, modality="fact", source="jev"),
        threshold=THRESHOLD,
    )
    assert not should_promote(
        RelationJudgment(assertion=0.99, modality="denial", source="jev"),
        threshold=THRESHOLD,
    )
    assert not should_promote(
        RelationJudgment(assertion=0.9, modality="forecast", source="jev"),
        threshold=THRESHOLD,
    )
    assert not should_promote(
        RelationJudgment(assertion=0.9, modality="not stated", source="jev"),
        threshold=THRESHOLD,
    )


def test_decide_uses_argmax_not_the_noul_cutoff() -> None:
    assert should_promote(
        RelationJudgment(assertion=0.2, modality="fact", agrees=True, source="decide"),
        threshold=THRESHOLD,
    )
    assert not should_promote(
        RelationJudgment(assertion=0.99, modality="fact", agrees=False, source="decide"),
        threshold=THRESHOLD,
    )
    assert should_promote(
        RelationJudgment(assertion=0.4, modality="fact", agrees=None, source="decide"),
        threshold=THRESHOLD,
    )


def test_jev_judge_asks_modality_beside_noul(monkeypatch) -> None:
    client = SystemOneClient(Config(systemone_api_key="test-key"))
    captured: dict = {}

    def fake_decide(state, questions):
        captured["questions"] = questions
        return {
            "answers": {
                "claim_7": {"type": "noul", "noul": 0.91},
                "modality_7": {
                    "type": "choice",
                    "choice": "forecast",
                    "probabilities": {"fact": 0.1, "forecast": 0.8},
                },
            }
        }

    monkeypatch.setattr(client, "decide", fake_decide)
    verdicts = judge_relations(client, [{
        "id": 7,
        "subject": "Nordwind Capital",
        "relation": "INVESTED_IN",
        "object": "GreenGrid Energy",
        "evidence": "Nordwind Capital will invest 400 million euros.",
    }])
    assert captured["questions"]["claim_7"]["type"] == "noul"
    assert captured["questions"]["modality_7"]["type"] == "choice"
    assert isinstance(captured["questions"]["modality_7"]["criteria"], dict)
    assert "fact" in captured["questions"]["modality_7"]["criteria"]
    assert verdicts[7].assertion == 0.91
    assert verdicts[7].modality == "forecast"
    assert verdicts[7].source == "jev"
    assert not should_promote(verdicts[7], threshold=THRESHOLD)


def test_decide_gate_with_a_stub_classifier() -> None:
    answers = {
        ("modality",): ("forecast", 0.7),
        ("relation",): ("no relation", 0.4),
    }

    def classify(text, task, labels, instruction=None):
        del text, labels, instruction
        return answers[(task,)]

    claims = [{
        "id": 3,
        "subject": "GreenGrid Energy",
        "relation": "LOCATED_IN",
        "object": "Poland",
        "evidence": "expansion into Poland",
        "source_type": "organization",
        "target_type": "location",
    }]
    verdicts = decide_judge(claims, classify=classify)
    assert verdicts[3].modality == "forecast"
    assert verdicts[3].agrees is False
    assert verdicts[3].source == "decide"
    assert not should_promote(verdicts[3], threshold=THRESHOLD)


def test_decide_skips_relation_check_when_types_are_unknown() -> None:
    def classify(text, task, labels, instruction=None):
        del text, labels, instruction
        assert task == "modality"
        return "fact", 0.8

    verdicts = decide_judge([{
        "id": 1,
        "subject": "A",
        "relation": "RELATED_TO",
        "object": "B",
        "evidence": "A is related to B.",
    }], classify=classify)
    assert verdicts[1].agrees is None
    assert should_promote(verdicts[1], threshold=THRESHOLD)
