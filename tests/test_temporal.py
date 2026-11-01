"""Fact validity dates read from evidence — pure, no DB. See DECISIONS.md D25."""

from __future__ import annotations

from datetime import date

from entity_graph.temporal import fact_validity

BIO = (
    "Mr. Rao served as Chief Financial Officer at Fanatics Commerce, an online "
    "sports merchandise and retail platform, from May 2023 to April 2024. Before "
    "that, he served as the Chief Financial Officer of Cedar, a healthcare "
    "technology company, from July 2021 to May 2023. From October 2015 to July "
    "2021, Mr. Rao held several key positions at Airbnb."
)


def test_each_role_gets_the_range_in_its_own_sentence() -> None:
    assert fact_validity(BIO, ["Fanatics Commerce"]) == (date(2023, 5, 1), date(2024, 4, 30))
    assert fact_validity(BIO, ["Cedar"]) == (date(2021, 7, 1), date(2023, 5, 31))
    # A leading date phrase, and "Mr." is not a sentence end.
    assert fact_validity(BIO, ["Airbnb"]) == (date(2015, 10, 1), date(2021, 7, 31))


def test_since_in_and_year_spans() -> None:
    assert fact_validity(
        "Dr. Narasimhan has served as the Chief Executive Officer of Novartis AG "
        "since February 2018.", ["Novartis AG"],
    ) == (date(2018, 2, 1), None)
    assert fact_validity(
        "In April 2026, the Company completed the acquisition of Coefficient Bio, Inc.",
        ["Coefficient Bio"],
    ) == (date(2026, 4, 1), None)
    assert fact_validity("He served as mayor (1928-1932) of Shanghai.", ["Shanghai"]) == (
        date(1928, 1, 1), date(1932, 12, 31),
    )


def test_years_inside_names_and_missing_anchors_give_no_date() -> None:
    assert fact_validity("The Securities Act of 1933 applies to Barclays.", ["Barclays"]) == (
        None, None,
    )
    assert fact_validity("Founded in 2021.", ["Nowhere Inc"]) == (None, None)
    assert fact_validity("", ["Cedar"]) == (None, None)


def test_a_date_does_not_jump_over_another_organization_or_clause() -> None:
    assert fact_validity(
        "Before joining OpenAI, Dr. Amodei served as a Senior Research Scientist at "
        "Google Brain, an AI research team at Google, from October 2015 to June 2016.",
        ["OpenAI"],
    ) == (None, None)
    assert fact_validity(
        "Mr. Liddell has served as a Director of Commonwealth Fusion Systems since "
        "March 2025, and as a Director of the Council on Foreign Relations.",
        ["Council on Foreign Relations"],
    ) == (None, None)
