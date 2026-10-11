"""Curation-file parsing tests — pure, no DB or network.

Merging and hiding are DB operations; only the plan format is unit-testable
offline. See DECISIONS.md D19.
"""

from __future__ import annotations

import json

import pytest

from entity_graph.curate import CurationSpec, load_curation


def _write(tmp_path, payload) -> str:
    path = tmp_path / "curation.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_load_full_spec(tmp_path) -> None:
    path = _write(tmp_path, {
        "renames": [["Soong", "T. V. Soong"]],
        "merges": [["T. V. Soong", "T. V. SOONG"], ["Tu Yueh-sheng", "Tu"]],
        "hide": ["central bank", "finance"],
        "unhide": ["commission"],
    })
    spec = load_curation(path)
    assert spec == CurationSpec(
        renames=[("Soong", "T. V. Soong")],
        merges=[("T. V. Soong", "T. V. SOONG"), ("Tu Yueh-sheng", "Tu")],
        hide=["central bank", "finance"],
        unhide=["commission"],
    )


def test_load_minimal_spec(tmp_path) -> None:
    spec = load_curation(_write(tmp_path, {}))
    assert spec == CurationSpec(renames=[], merges=[], hide=[], unhide=[])


def test_load_rejects_bad_merge_pair(tmp_path) -> None:
    path = _write(tmp_path, {"merges": [["only-one-name"]]})
    with pytest.raises(ValueError):
        load_curation(path)


def test_load_rejects_bad_rename_pair(tmp_path) -> None:
    path = _write(tmp_path, {"renames": [["only-one-name"]]})
    with pytest.raises(ValueError):
        load_curation(path)
