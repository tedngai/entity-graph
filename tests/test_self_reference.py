"""Self-reference detection, rewriting and span mapping — pure. See DECISIONS.md D28."""

from __future__ import annotations

from entity_graph.self_reference import (
    find_self_reference,
    rewrite,
    term_map,
)

COREWEAVE = ("Unless otherwise indicated, the terms “CoreWeave,” the “Company,” “we,” "
             "“us,” and “our” refer to CoreWeave, Inc. and our subsidiaries.")
ANTHROPIC = ("Unless the context otherwise requires, the terms “Anthropic,” “the company,”\n\n"
             "“we,” “us,” and “our” in this prospectus refer to Anthropic, PBC and its "
             "consolidated subsidiaries, and")


def test_finds_both_filing_definitions() -> None:
    cw = find_self_reference(COREWEAVE)
    assert (cw.name, cw.short) == ("CoreWeave, Inc.", "CoreWeave")
    an = find_self_reference(ANTHROPIC)
    assert (an.name, an.short) == ("Anthropic, PBC", "Anthropic")


def test_definition_with_words_before_the_name() -> None:
    # CAVA S-1 wording: "mean the business of" kept every "we" unresolved (D37).
    ref = find_self_reference("•the “Company,” “we,” “us,” and “our” mean the business of "
                              "CAVA Group, Inc. and its subsidiaries;")
    assert (ref.name, ref.short) == ("CAVA Group, Inc.", "CAVA Group")
    ref = find_self_reference("the terms “we,” “us,” and “our” refer to the combined "
                              "business of Acme Holdings, Inc. and its subsidiaries")
    assert ref.name == "Acme Holdings, Inc."
    # Lowercase words that never reach a capitalized name still match nothing.
    assert find_self_reference("“we,” “us” and “our” mean the people who wrote this") is None


def test_documents_without_a_definition_are_left_alone() -> None:
    assert find_self_reference("We met GreenGrid in Berlin. Our view is that it grew.") is None
    # Quoted defined terms that do not include we/us/our are D24's business.
    assert find_self_reference('the terms “Notes” and “Indenture” refer to the 2030 Notes') is None


def test_rewrite_names_the_author_and_maps_spans_back() -> None:
    ref = find_self_reference(COREWEAVE)
    text = "Nitin Agrawal has served as our Chief Financial Officer. We acquired Conductor. US law."
    rw = rewrite(text, ref)
    assert rw.text == ("Nitin Agrawal has served as CoreWeave’s Chief Financial Officer. "
                       "CoreWeave acquired Conductor. US law.")
    name = rw.text.index("CoreWeave")
    start, end = rw.span(name, name + len("CoreWeave"))
    assert text[start:end] == "our"
    chief = rw.text.index("Chief")
    start, end = rw.span(chief, chief + len("Chief Financial Officer"))
    assert text[start:end] == "Chief Financial Officer"
    law = rw.text.index("US law")
    assert text[slice(*rw.span(law, law + 6))] == "US law"


def test_our_company_is_the_author_and_the_company_is_replaced_when_defined() -> None:
    ref = find_self_reference(COREWEAVE)
    rw = rewrite("Microsoft made investments in our company. The Company paid it.", ref)
    assert rw.text == "Microsoft made investments in CoreWeave. CoreWeave paid it."


def test_term_map() -> None:
    ref = find_self_reference(COREWEAVE)
    terms = term_map(ref)
    assert terms["we"] == terms["the Company"] == terms["CoreWeave"] == "CoreWeave, Inc."
    assert "US" not in terms


def test_window_keeps_both_readings_with_original_offsets_and_evidence() -> None:
    from entity_graph.extract.base import ExtractionResult, Mention, RelationProposal
    from entity_graph.pipeline import _extract_window

    class Fake:
        """Finds 'CoreWeave' only where the text spells it out."""

        def extract(self, text: str, doc_key: str = "") -> ExtractionResult:
            mentions = [Mention("Nitin Agrawal", "PERSON", 0, 13, 0.9)]
            relations = []
            if "CoreWeave" in text:
                i = text.index("CoreWeave")
                mentions.append(Mention("CoreWeave", "ORG", i, i + 9, 0.9))
                relations.append(RelationProposal(
                    "Nitin Agrawal", "CoreWeave", "WORKS_FOR", text, 0.9, "person", "organization"))
            return ExtractionResult(doc_key, mentions, relations)

    ref = find_self_reference(COREWEAVE)
    text = "Nitin Agrawal has served as our Chief Financial Officer."
    result = _extract_window(Fake(), text, "", ref)
    (relation,) = result.relations
    assert (relation.target_surface, relation.evidence) == ("CoreWeave", text)
    spans = {(m.surface, m.start, m.end) for m in result.mentions}
    assert ("our", 28, 31) in spans and ("Nitin Agrawal", 0, 13) in spans
    assert len(result.mentions) == len(spans) == 2
    assert _extract_window(Fake(), text, "", None).relations == []
