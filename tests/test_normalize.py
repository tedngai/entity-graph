"""Surface-normalization tests — pure functions, no DB or network.

Normalization lets near-duplicate surfaces (dates especially) collapse to one
node instead of spawning one per surface. See DECISIONS.md D18.
"""

from entity_graph.resolve import (
    _normalize_name,
    canonical_date,
    match_surface,
    normalize_surface,
)


def test_normalize_surface_collapses_whitespace_and_trims_punctuation() -> None:
    assert normalize_surface("T. V.\n  Soong") == "T. V. Soong"
    assert normalize_surface("  Shanghai  ") == "Shanghai"
    assert normalize_surface("commission.") == "commission"
    assert normalize_surface("(Green Gang)") == "Green Gang"
    assert normalize_surface("") == ""


def test_canonical_date() -> None:
    assert canonical_date("March 1928") == "1928"
    assert canonical_date("1928-1932") == "1928"
    assert canonical_date("December 1930-August 1931") == "1930"
    assert canonical_date("no year here") is None


def test_match_surface_collapses_dates() -> None:
    assert match_surface("1928-1932", "DATE") == "1928"
    assert match_surface("March 1928", "DATE") == "1928"
    assert match_surface("1928", "DATE") == "1928"
    # Same key for a range and a month variant -> one node.
    assert match_surface("1928-1931", "DATE") == match_surface("March 1928", "DATE")


def test_match_surface_keeps_non_dates_but_normalizes() -> None:
    assert match_surface("T. V.\nSoong", "PERSON") == "T. V. Soong"
    assert match_surface("GreenGrid Energy.", "ORG") == "GreenGrid Energy"
    # A DATE-typed surface with no year is left normalized.
    assert match_surface("mid-February", "DATE") == "mid-February"


def test_normalize_name_alias() -> None:
    assert _normalize_name("  T. V.\nSoong ") == "T. V. Soong"
