"""Defined-term detection — pure, no DB. See DECISIONS.md D24."""

from __future__ import annotations

from entity_graph.defined_terms import defined_term_map, find_defined_terms


def _mentions(text: str, *surfaces: str, entity_type: str = "ORG") -> list[tuple]:
    out = []
    for surface in surfaces:
        start = text.index(surface)
        out.append((start, start + len(surface), surface, entity_type))
    return out


def test_quoted_terms_link_to_the_preceding_mention() -> None:
    text = (
        "In 2023, the Company established the Long-Term Benefit Trust (“LTBT”). "
        "Anthropic, PBC (the “Company”) is a public benefit corporation. "
        "Amazon.com, Inc. (together with its affiliates, “Amazon”) invested."
    )
    terms = defined_term_map(
        text, _mentions(text, "Long-Term Benefit Trust", "Anthropic, PBC", "Amazon.com, Inc.")
    )
    assert terms["LTBT"].long == "Long-Term Benefit Trust"
    assert terms["Company"].long == "Anthropic, PBC"
    assert terms["Amazon"].long == "Amazon.com, Inc."


def test_terms_that_name_something_new_are_not_aliases() -> None:
    text = (
        "Bartz, et al. v. Anthropic PBC (“Bartz”) settled. "
        "We issued notes to Amazon (“Amazon 2024 NPA”). "
        "Accounting principles generally accepted in the United States (“GAAP”)."
    )
    terms = defined_term_map(text, _mentions(text, "Anthropic PBC", "Amazon", "United States"))
    assert terms == {}


def test_bare_parentheses_need_an_acronym_of_the_long_form() -> None:
    text = (
        "The British and American Tobacco Company (B.A.T.) sold. "
        "Adjusted Operating Income (Loss) fell. "
        "Dr. Shah led the Clinton Health Access Initiative (CHAI)."
    )
    terms = {t.short: t.long for t in find_defined_terms(
        text, _mentions(text, "Adjusted Operating Income")
    )}
    assert terms["B.A.T"] == "British and American Tobacco Company"
    assert terms["CHAI"] == "Clinton Health Access Initiative"
    assert "Loss" not in terms


def test_pronouns_are_never_defined_terms() -> None:
    text = "Anthropic, PBC (“we” or “us”) builds models."
    assert defined_term_map(text, _mentions(text, "Anthropic, PBC")) == {}
