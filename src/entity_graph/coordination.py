"""Coordination: a relation to one member of a list holds for the list (D30).

"for whom Morgan Stanley & Co. LLC, J.P. Morgan Securities LLC, and Goldman
Sachs & Co. LLC are acting as representatives" — JointIE links one or two of
the three. When a relation endpoint is one of several same-type mentions
joined only by commas, "and" or "or", the relation is proposed for each
member. The gate still judges every copy against the same evidence.
"""

from __future__ import annotations

import re
from dataclasses import replace

from .extract.base import ExtractionResult, Mention

# What may separate two list members: ", " / ", and " / " and " / " or ".
_JOINER = re.compile(r"\s*(?:,\s*(?:(?:and|or)\s+)?|(?:and|or)\s+)")
# A copy is a little less certain than the relation the model emitted.
COPY_CONFIDENCE = 0.9


def _non_overlapping(mentions: list[Mention]) -> list[Mention]:
    """Longest mention wins where spans overlap."""
    chosen: list[Mention] = []
    for m in sorted(mentions, key=lambda m: (m.start, -(m.end - m.start))):
        if chosen and m.start < chosen[-1].end:
            continue
        chosen.append(m)
    return chosen


def coordinated_groups(text: str, mentions: list[Mention]) -> list[list[Mention]]:
    """Runs of two or more same-type mentions joined only by list punctuation."""
    groups: list[list[Mention]] = []
    run: list[Mention] = []
    for m in _non_overlapping(mentions):
        if run and m.entity_type == run[-1].entity_type and _JOINER.fullmatch(
            text[run[-1].end:m.start]
        ):
            run.append(m)
            continue
        if len(run) > 1:
            groups.append(run)
        run = [m]
    if len(run) > 1:
        groups.append(run)
    return groups


def expand_coordination(result: ExtractionResult, text: str) -> ExtractionResult:
    """Copy each relation onto the other members of an endpoint's list."""
    groups = coordinated_groups(text, result.mentions)
    if not groups:
        return result
    member_of: dict[str, list[str]] = {}
    for group in groups:
        surfaces = [m.surface for m in group]
        for surface in surfaces:
            member_of.setdefault(surface, surfaces)
    seen = {(r.source_surface, r.target_surface, r.relation) for r in result.relations}
    added = []
    for proposal in result.relations:
        for side in ("source_surface", "target_surface"):
            members = member_of.get(getattr(proposal, side))
            if not members:
                continue
            other = proposal.target_surface if side == "source_surface" else proposal.source_surface
            if other in members:
                continue  # a relation inside the list is not about the list
            for member in members:
                copy = replace(proposal, **{side: member},
                               confidence=proposal.confidence * COPY_CONFIDENCE)
                key = (copy.source_surface, copy.target_surface, copy.relation)
                if key not in seen:
                    seen.add(key)
                    added.append(copy)
    if not added:
        return result
    return replace(result, relations=[*result.relations, *added])
