"""Defined terms: short names a document declares for a longer name.

Filings, contracts, and scholarship declare abbreviations inline::

    Long-Term Benefit Trust ("LTBT")
    Anthropic, PBC (the "Company")
    Amazon.com, Inc. (together with its affiliates, "Amazon")
    British and American Tobacco Company (B.A.T.)

Resolution reads these so every later mention of the short form links to the
entity of the long form, instead of leaving ``LTBT`` and ``Long-Term Benefit
Trust`` as two nodes. A definition is scoped to its own document: nothing is
written to ``entity_aliases``, so ``the "Company"`` of one filing never
captures another document's ``Company``. See DECISIONS.md D24.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ("X"), (the "X"), (each, a "X"), (together with its affiliates, "X").
_QUOTED = re.compile(
    r"\(\s*(?:[^()\"“”\n]{0,60},\s*)?(?:the |each, an? |an? )?[\"“]([^\"“”\n]{1,40})[\"”]\s*\)"
)
# (B.A.T.), (CHAI) — accepted only when the letters match the preceding words.
_BARE = re.compile(r"\(\s*([A-Z][A-Za-z.&]{1,11})\s*\)")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’]*")
_GAP = re.compile(r"[\s,]*")
_MINOR = {
    "and", "of", "the", "for", "on", "in", "&", "de", "du", "la",
    "a", "an", "company", "co", "inc", "corporation", "corp", "llc", "ltd", "plc", "group",
}
# Conventional defined names for the party itself: Anthropic, PBC (the
# "Company"). These may stand for any organization name.
_PARTY_NAMES = {
    "company", "corporation", "registrant", "issuer", "trust", "fund",
    "partnership", "bank", "firm", "society",
}
# Short forms that are never names, even when a document quotes them.
_NOT_NAMES = {"we", "us", "our", "it", "its", "they", "you", "your"}


@dataclass(frozen=True)
class DefinedTerm:
    short: str
    long: str
    long_type: str | None
    position: int  # char offset of the parenthetical


def _letters(token: str) -> str:
    return "".join(ch for ch in token if ch.isalpha()).upper()


def _spells(target: str, parts: list[tuple[str, bool]]) -> bool:
    """True when ``target`` is the initials of ``parts``, skipping minor words."""
    def walk(i: int, j: int) -> bool:
        if j == len(parts):
            return i == len(target)
        letter, minor = parts[j]
        if i < len(target) and letter == target[i] and walk(i + 1, j + 1):
            return True
        return minor and walk(i, j + 1)

    return walk(0, 0)


def _initials_match(acronym: str, words: list[str]) -> bool:
    """True when the acronym spells the initials of a trailing run of ``words``.

    Minor words (and, of, the) and corporate suffixes (Company, Inc.) may be
    skipped; hyphenated words contribute each part (Long-Term -> L, T).
    """
    target = _letters(acronym)
    if len(target) < 2:
        return False
    parts: list[tuple[str, bool]] = []
    for word in words:
        for piece in re.split(r"[-‐]", word):
            if piece:
                parts.append((piece[0].upper(), piece.lower().rstrip(".") in _MINOR))
    for start in range(len(parts) - 1, -1, -1):
        if not parts[start][1] and _spells(target, parts[start:]):
            return True
    return False


def _is_acronym(short: str) -> bool:
    core = short[:-1] if short.endswith("s") and short[:-1].isupper() else short
    letters = core.replace(".", "").replace("&", "")
    return len(letters) >= 2 and letters.isupper() and " " not in core


def _compatible(short: str, long: str, long_type: str | None) -> bool:
    """Whether ``short`` can be a name for ``long`` rather than a new thing.

    A quoted term often names something the sentence defines (a contract, a
    case, a date) right after an unrelated noun: Bartz, et al. v. Anthropic PBC
    ("Bartz") or the Amazon note purchase agreement ("Amazon 2024 NPA"). So:
    an acronym must spell the long form's initials; a word term must be made
    of the long form's own words, or be a conventional party name ("Company")
    for an organization.
    """
    if _is_acronym(short):
        core = short[:-1] if short.endswith("s") and short[:-1].isupper() else short
        return _initials_match(core, _WORD.findall(long))
    short_words = {w.casefold() for w in _WORD.findall(short)}
    long_words = {w.casefold() for w in _WORD.findall(long)}
    if short_words and short_words <= long_words:
        return True
    return short.casefold() in _PARTY_NAMES and (long_type or "").upper() == "ORG"


def _acronym_long_form(text: str, paren_start: int, acronym: str) -> str | None:
    """The run of preceding words whose initials spell ``acronym``."""
    window = text[max(0, paren_start - 160):paren_start]
    words = _WORD.findall(window)
    target_len = len(_letters(acronym))
    for size in range(target_len, min(len(words), target_len * 2) + 1):
        tail = words[-size:]
        if tail and tail[0].lower() not in _MINOR and _initials_match(acronym, tail):
            first = window.rfind(tail[0], 0, len(window) - len(" ".join(tail[1:])) + 1)
            if first < 0:
                return None
            return " ".join(window[first:].split()).rstrip(" ,")
    return None


def find_defined_terms(
    text: str, mentions: list[tuple[int, int, str, str | None]]
) -> list[DefinedTerm]:
    """Definitions in ``text``, anchored to the mention just before each one.

    ``mentions`` are ``(char_start, char_end, surface, entity_type)`` for the
    same document. The long form is the extracted mention ending right before
    the parenthetical (only spaces or a comma between). Without one, an
    acronym falls back to the preceding words whose initials spell it.
    """
    by_end: dict[int, tuple[str, str | None]] = {}
    for start, end, surface, entity_type in mentions:
        previous = by_end.get(end)
        if previous is None or len(surface) > len(previous[0]):
            by_end[end] = (surface, entity_type)

    def mention_before(position: int) -> tuple[str, str | None] | None:
        lo = max(0, position - 4)
        for end in range(position, lo - 1, -1):
            hit = by_end.get(end)
            if hit and _GAP.fullmatch(text[end:position]):
                return hit
        return None

    terms: list[DefinedTerm] = []
    candidates = [(m.start(), m.group(1), True) for m in _QUOTED.finditer(text)]
    candidates += [(m.start(), m.group(1), False) for m in _BARE.finditer(text)]
    for position, short, quoted in sorted(candidates):
        short = " ".join(short.split()).strip(" ,.;:")
        if not short or short.lower() in _NOT_NAMES:
            continue
        long_hit = mention_before(position)
        if long_hit is not None:
            long_surface, long_type = long_hit
            # A bare parenthetical counts only as an acronym of the long form;
            # "(Loss)" or "(Continued)" never does.
            if not quoted and not _is_acronym(short):
                continue
            if not _compatible(short, long_surface, long_type):
                continue
        else:
            if not _is_acronym(short):
                continue
            fallback = _acronym_long_form(text, position, short)
            if fallback is None:
                continue
            long_surface, long_type = fallback, None
        long_surface = " ".join(long_surface.split())
        if long_surface.casefold() == short.casefold():
            continue
        terms.append(DefinedTerm(short, long_surface, long_type, position))
    return terms


def defined_term_map(
    text: str, mentions: list[tuple[int, int, str, str | None]]
) -> dict[str, DefinedTerm]:
    """Short form (exact case, whitespace-normalized) -> its first definition."""
    mapping: dict[str, DefinedTerm] = {}
    for term in find_defined_terms(text, mentions):
        mapping.setdefault(term.short, term)
    return mapping
