"""JointIE payload parsing. No model download."""

from entity_graph.extract.gliner_extractor import parse_joint_payload

TEXT = "Maria Chen founded GreenGrid Energy in Berlin."


def _payload() -> dict:
    return {
        "entities": [
            {"id": "e1", "type": "person", "text": "Maria Chen",
             "start": 0, "end": 10, "confidence": 0.9},
            {"id": "e2", "type": "organization", "text": "GreenGrid Energy",
             "start": 19, "end": 35, "confidence": 0.8},
            {"id": "e3", "type": "location", "text": "Berlin",
             "start": 39, "end": 45, "confidence": 0.7},
        ],
        "relations": [
            {"type": "founded", "head": "e1", "tail": "e2", "confidence": 0.88},
            {"type": "located_in", "head": "e2", "tail": "e3", "confidence": 0.6},
            {"type": "works_for", "head": "e1", "tail": "e1", "confidence": 0.99},
        ],
    }


def test_joint_payload_keeps_typed_endpoints() -> None:
    result = parse_joint_payload(_payload(), TEXT, doc_key="sample")
    surfaces = {mention.surface: mention for mention in result.mentions}
    assert surfaces["Maria Chen"].entity_type == "PERSON"
    assert surfaces["Maria Chen"].start == 0
    assert TEXT[surfaces["GreenGrid Energy"].start:surfaces["GreenGrid Energy"].end] == (
        "GreenGrid Energy"
    )
    founded = next(rel for rel in result.relations if rel.relation == "FOUNDED")
    assert founded.source_surface == "Maria Chen"
    assert founded.target_surface == "GreenGrid Energy"
    assert founded.source_type == "person"
    assert founded.target_type == "organization"
    assert "founded" in founded.evidence
    assert all(rel.source_surface != rel.target_surface for rel in result.relations)
    assert not any(rel.relation == "WORKS_FOR" for rel in result.relations)


def test_infeasible_joint_result_drops_relations() -> None:
    result = parse_joint_payload(_payload(), TEXT, feasible=False, doc_key="sample")
    assert result.mentions
    assert result.relations == []
