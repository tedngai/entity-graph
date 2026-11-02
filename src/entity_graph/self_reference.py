"""Self-reference: the entity a document calls "we", "us", "our" (D28).

Corporate documents write about their author in the first person and define
it once::

    the terms "CoreWeave," the "Company," "we," "us," and "our" refer to
    CoreWeave, Inc. and our subsidiaries.

Everything after that sentence says "our Chief Financial Officer" or "we
acquired Conductor", and the extractor rarely tags a pronoun as an
organization: in the CoreWeave S-1 it tagged 543 of about 6,500 such words,
so most facts about the author were never proposed.

``find_self_reference`` reads the definition. ``rewrite`` swaps the defined
terms for the author's name in a window before extraction and maps spans
back, so mentions keep their original offsets and surfaces and evidence stays
a verbatim quote. ``term_map`` lets resolution link the pronoun mentions to
the author. A document without such a definition is left alone: in a history
or a memo "we" may be the writer, not a party.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass

_QUOTE = r"[\"“]([^\"“”\n]{1,40}?)[,.]?[\"”]"
_SEP = r"[\s,]*(?:(?:and|or)\s+)?(?:the\s+)?"
# A run of two or more quoted terms, then "refer(s) to" / "mean(s)" and a name.
_DEFINITION = re.compile(
    rf"(?P<terms>(?:{_QUOTE}{_SEP}){{2,}})"
    r"[^.;\n\"“”]{0,60}?\b(?:refers?\s+to|means?)\s+"
    r"(?P<name>[A-Z][\w&’'.-]*(?:,?\s+(?:[A-Z][\w&’'.-]*|Inc\.?|LLC|PBC|plc|L\.P\.))*)"
)
_TERM = re.compile(_QUOTE)
_PRONOUNS = {"we", "us", "our"}
_PARTY_NAMES = {"company", "the company", "registrant", "issuer", "corporation"}


@dataclass(frozen=True)
class SelfReference:
    name: str                # the defined entity: "CoreWeave, Inc."
    short: str               # name written into rewritten text: "CoreWeave"
    terms: tuple[str, ...]   # the quoted terms, as defined
    position: int            # char offset of the definition


def find_self_reference(text: str) -> SelfReference | None:
    """The first definition whose quoted terms include we / us / our."""
    for match in _DEFINITION.finditer(text):
        terms = [t.strip().strip(",") for t in _TERM.findall(match.group("terms"))]
        if not _PRONOUNS & {t.lower() for t in terms}:
            continue
        name = match.group("name").rstrip(" ,")
        if name.endswith(".") and not re.search(r"(?:Inc|L\.P|Co|Corp|Ltd)\.$", name):
            name = name[:-1]
        proper = [t for t in terms if t[:1].isupper() and t.lower() not in _PARTY_NAMES]
        short = proper[0] if proper else name.split(",")[0].strip()
        return SelfReference(name=name, short=short, terms=tuple(terms),
                             position=match.start())
    return None


def _patterns(ref: SelfReference) -> list[tuple[re.Pattern, str]]:
    """(pattern, replacement) pairs, longest first. Pronouns match lowercase or
    sentence-initial capitals only, so "US" (the country) is never touched."""
    possessive = f"{ref.short}’s"
    pairs: list[tuple[re.Pattern, str]] = [
        # "our company" is the author itself, not something it owns.
        (re.compile(r"\b[Oo]ur\s+(?:[Cc]ompany|[Cc]orporation)\b"), ref.short),
        (re.compile(r"\b(?:[Oo]ur|[Oo]urs)\b"), possessive),
        (re.compile(r"\b(?:[Ww]e|us|[Oo]urselves)\b"), ref.short),
    ]
    party = {t.lower() for t in ref.terms} & _PARTY_NAMES
    if party:
        words = sorted({p.removeprefix("the ") for p in party}, key=len, reverse=True)
        alternation = "|".join(re.escape(w) for w in words)
        pairs.insert(0, (re.compile(rf"\b[Tt]he\s+(?:{alternation.title()}|{alternation})\b"),
                         ref.short))
    return pairs


@dataclass(frozen=True)
class Rewrite:
    text: str
    # Sorted (rewritten_start, rewritten_end, original_start, original_end).
    edits: tuple[tuple[int, int, int, int], ...]

    def to_original(self, position: int, *, end: bool = False) -> int:
        """Map a rewritten offset to the original text. An offset inside an
        inserted name maps to the start (or, for ``end``, the end) of the
        word it replaced."""
        starts = [e[0] for e in self.edits]
        i = bisect_right(starts, position) - 1
        if i < 0:
            return position
        r_start, r_end, o_start, o_end = self.edits[i]
        if position < r_end or (end and position == r_end):
            if position == r_start and not end:
                return o_start
            return o_end if end else o_start
        return position - r_end + o_end

    def span(self, start: int, end: int) -> tuple[int, int]:
        return self.to_original(start), self.to_original(end, end=True)


def rewrite(text: str, ref: SelfReference) -> Rewrite:
    """Replace the defined terms in ``text`` with ``ref.short``."""
    hits: list[tuple[int, int, str]] = []
    taken: list[tuple[int, int]] = []
    for pattern, replacement in _patterns(ref):
        for m in pattern.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            hits.append((m.start(), m.end(), replacement))
            taken.append((m.start(), m.end()))
    hits.sort()
    out: list[str] = []
    edits: list[tuple[int, int, int, int]] = []
    cursor = 0
    written = 0
    for start, end, replacement in hits:
        out.append(text[cursor:start])
        written += start - cursor
        edits.append((written, written + len(replacement), start, end))
        out.append(replacement)
        written += len(replacement)
        cursor = end
    out.append(text[cursor:])
    return Rewrite(text="".join(out), edits=tuple(edits))


def term_map(ref: SelfReference) -> dict[str, str]:
    """Surfaces (exact case) that resolve to ``ref.name``."""
    surfaces = {"we", "We", "us", "our", "Our", "ours", "ourselves"}
    for term in ref.terms:
        if term.lower() in _PRONOUNS:
            continue
        surfaces.update({term, term[:1].upper() + term[1:]})
        if term.lower() in _PARTY_NAMES:
            bare = term.removeprefix("the ").removeprefix("The ")
            surfaces.update({bare, bare.title(), f"the {bare}", f"The {bare}",
                             f"the {bare.title()}", f"The {bare.title()}"})
    surfaces.discard(ref.name)
    return {surface: ref.name for surface in surfaces}

