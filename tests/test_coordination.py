"""Coordinated lists — pure, no model. See DECISIONS.md D30."""

from __future__ import annotations

from entity_graph.coordination import coordinated_groups, expand_coordination
from entity_graph.extract.base import ExtractionResult, Mention, RelationProposal

TEXT = ("for whom Morgan Stanley & Co. LLC, J.P. Morgan Securities LLC, and Goldman "
        "Sachs & Co. LLC are acting as representatives of the underwriters for CoreWeave.")


def _mention(surface: str, entity_type: str = "ORG") -> Mention:
    start = TEXT.index(surface)
    return Mention(surface, entity_type, start, start + len(surface), 0.9)


MENTIONS = [
    _mention("Morgan Stanley & Co. LLC"),
    _mention("Morgan Stanley"),  # nested; the longer span wins
    _mention("J.P. Morgan Securities LLC"),
    _mention("Goldman Sachs & Co. LLC"),
    _mention("CoreWeave"),
]


def test_groups_join_only_on_list_punctuation() -> None:
    (group,) = coordinated_groups(TEXT, MENTIONS)
    assert [m.surface for m in group] == [
        "Morgan Stanley & Co. LLC", "J.P. Morgan Securities LLC", "Goldman Sachs & Co. LLC",
    ]


def test_relation_on_one_member_is_copied_to_the_others() -> None:
    proposal = RelationProposal("Goldman Sachs & Co. LLC", "CoreWeave", "UNDERWRITER_OF",
                                TEXT, 0.8, "organization", "organization")
    result = expand_coordination(ExtractionResult("", MENTIONS, [proposal]), TEXT)
    sources = sorted(r.source_surface for r in result.relations)
    assert sources == ["Goldman Sachs & Co. LLC", "J.P. Morgan Securities LLC",
                       "Morgan Stanley & Co. LLC"]
    assert all(r.evidence == TEXT for r in result.relations)
    assert min(r.confidence for r in result.relations) < 0.8


def test_relations_inside_a_list_and_mixed_types_are_not_copied() -> None:
    inside = RelationProposal("Morgan Stanley & Co. LLC", "Goldman Sachs & Co. LLC",
                              "PARTNERED_WITH", TEXT, 0.8)
    result = expand_coordination(ExtractionResult("", MENTIONS, [inside]), TEXT)
    assert len(result.relations) == 1
    mixed = [_mention("Morgan Stanley & Co. LLC"), _mention("J.P. Morgan Securities LLC", "PERSON")]
    assert coordinated_groups(TEXT, mixed) == []
