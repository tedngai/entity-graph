"""Validity dates for a fact, read from its evidence quote.

DATE is not a node type (D18), so dates never reach the graph as entities;
before this module no fact carried ``valid_from`` / ``valid_to`` even though
biographies state them ("Chief Financial Officer of Cedar ... from July 2021
to May 2023"). The rule is deliberately narrow:

* only prepositional date expressions count — ``from X to Y``, ``between X
  and Y``, ``since X``, ``until X``, ``in X``, ``as of X`` — so a year inside a
  name (``Securities Act of 1933``) is never read as a date;
* the expression must sit in the sentence that names the fact's object: the
  first one after the object wins (bios list role, then dates), else the last
  one before it — unless another organization (``at Google Brain``) or a new
  clause (``, and as a Director of``) sits between them.

Month precision is kept; a year alone means January 1 (start) or December 31
(end). See DECISIONS.md D25.
"""

from __future__ import annotations

import calendar
import re
from datetime import date

_MONTHS = {
    name.lower(): index
    for index, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update({
    name.lower(): index for index, name in enumerate(calendar.month_abbr) if name
})
_MONTHS["sept"] = 9

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|" \
         r"November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\.?"
_YEAR = r"(?:1[5-9]\d{2}|20\d{2})(?!\d|,\d)"
_DATE = rf"(?:{_MONTH}\s+(?:\d{{1,2}},?\s+)?{_YEAR}|{_YEAR})"
# A period after these is not a sentence end ("Mr. Rao", "Coefficient Bio, Inc.").
_ABBREVIATIONS = {
    "mr", "ms", "mrs", "dr", "prof", "st", "jr", "sr", "inc", "corp", "co", "ltd",
    "no", "vs", "v", "u.s", "e.g", "i.e", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
}
_BOUNDARY = re.compile(r"[.!?][\"”’)]*\s+")
# Text between the fact and a date that ties the date to something else: another
# organization ("at Google Brain") or a new clause ("..., and as a Director of").
_BLOCKERS = re.compile(
    r"\b(?:at|with|joining|joined|of)\s+(?:the\s+)?[A-Z][\w&.'’-]*|;|,\s+and\b|\band\s+as\b"
)
_OPEN_END = r"(?:the\s+)?present|today|date"

_EXPRESSIONS = re.compile(
    rf"\b(?:"
    rf"from\s+(?P<range_a>{_DATE})\s+(?:to|until|through|-|–)\s+(?:(?P<range_b>{_DATE})|{_OPEN_END})"
    rf"|between\s+(?P<btw_a>{_DATE})\s+and\s+(?P<btw_b>{_DATE})"
    rf"|(?P<span_a>{_YEAR})\s*[-–]\s*(?P<span_b>{_YEAR})"
    rf"|since\s+(?P<since>{_DATE})"
    rf"|(?:until|through)\s+(?P<until>{_DATE})"
    rf"|(?:in|on|as\s+of|during)\s+(?P<point>{_DATE})"
    rf")",
    re.IGNORECASE,
)


def _parse(text: str, *, end: bool) -> date | None:
    """A date expression -> its first (or, with ``end``, last) calendar day."""
    year_match = re.search(r"(1[5-9]\d{2}|20\d{2})", text)
    if not year_match:
        return None
    year = int(year_match.group(1))
    month_match = re.match(r"([A-Za-z]+)", text.strip())
    month = _MONTHS.get(month_match.group(1).lower()) if month_match else None
    if month is None:
        return date(year, 12, 31) if end else date(year, 1, 1)
    if end:
        return date(year, month, calendar.monthrange(year, month)[1])
    return date(year, month, 1)


def _validity(match: re.Match) -> tuple[date | None, date | None]:
    groups = match.groupdict()
    if groups["range_a"]:
        to = _parse(groups["range_b"], end=True) if groups["range_b"] else None
        return _parse(groups["range_a"], end=False), to
    if groups["btw_a"]:
        return _parse(groups["btw_a"], end=False), _parse(groups["btw_b"], end=True)
    if groups["span_a"]:
        return _parse(groups["span_a"], end=False), _parse(groups["span_b"], end=True)
    if groups["since"]:
        return _parse(groups["since"], end=False), None
    if groups["until"]:
        return None, _parse(groups["until"], end=True)
    return _parse(groups["point"], end=False), None


def _sentences(text: str) -> list[tuple[int, int]]:
    """Sentence spans that do not break after honorifics, initials, or Inc."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        token = re.search(r"([A-Za-z.]+)[.!?][\"”’)]*$", text[start:match.end()].rstrip())
        word = token.group(1).lower().rstrip(".") if token else ""
        if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
            continue
        spans.append((start, match.end()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def _locate(text: str, names: list[str]) -> tuple[int, int] | None:
    lowered = text.casefold()
    for name in sorted({n for n in names if n}, key=len, reverse=True):
        index = lowered.find(name.casefold())
        if index >= 0:
            return index, index + len(name)
    return None


def fact_validity(
    evidence: str, objects: list[str], subjects: list[str] = ()
) -> tuple[date | None, date | None]:
    """``(valid_from, valid_to)`` for a fact, or ``(None, None)``.

    ``objects`` / ``subjects`` are names (surface, canonical name, aliases)
    used to find the fact in its evidence; the object is preferred because a
    biography sentence ties dates to the organization, not the person.
    """
    if not evidence:
        return None, None
    anchor = _locate(evidence, list(objects)) or _locate(evidence, list(subjects))
    if anchor is None:
        return None, None
    for start, end in _sentences(evidence):
        if start <= anchor[0] < end:
            sentence_start, sentence = start, evidence[start:end]
            break
    else:
        return None, None
    a_start, a_end = anchor[0] - sentence_start, anchor[1] - sentence_start
    matches = list(_EXPRESSIONS.finditer(sentence))
    after = [m for m in matches if m.start() >= a_end]
    before = [m for m in matches if m.end() <= a_start]
    if after and not _BLOCKERS.search(sentence[a_end:after[0].start()]):
        return _validity(after[0])
    if before and not _BLOCKERS.search(sentence[before[-1].end():a_start]):
        return _validity(before[-1])
    return None, None
