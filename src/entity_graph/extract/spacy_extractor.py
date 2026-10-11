"""spaCy extractor backend — the lightweight fallback (PROGRESS P4.3).

Runs a local spaCy pipeline for NER plus dependency-based relation extraction.
It is the fast, no-GPU alternative to GLiNER2 when a document is short, the
model is unavailable, or a second opinion is useful for type arbitration.

Requires the extraction extra and a model::

    pip install -e .[extraction]
    python -m spacy download en_core_web_sm
    set EXTRACTION_BACKEND=spacy

Relations come from the dependency tree: for each pair of entities in a
sentence we find the lowest common ancestor and, when it is a verb, map the
lemma onto our relation vocabulary (``found`` -> ``FOUNDED`` etc.). The verb's
subject becomes the source, its object the target. Unmapped verbs fall back to
``RELATED_TO``; pairs with no governing verb are dropped.
"""

from __future__ import annotations

import logging
from itertools import combinations

from .base import ExtractionResult, Mention, RelationProposal

log = logging.getLogger(__name__)

# spaCy NER labels -> our entity vocabulary (see extract.base.ENTITY_TYPES).
LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "PLACE",
    "LOC": "PLACE",
    "FAC": "PLACE",
    "EVENT": "EVENT",
    "DATE": "DATE",
    "TIME": "DATE",
    "PRODUCT": "CONCEPT",
    "WORK_OF_ART": "CONCEPT",
    "LAW": "CONCEPT",
    "LANGUAGE": "CONCEPT",
    "NORP": "CONCEPT",
}

# Verb lemma -> relation name. Unlisted verbs become RELATED_TO.
VERB_RELATIONS = {
    "found": "FOUNDED",
    "cofound": "FOUNDED",
    "establish": "FOUNDED",
    "launch": "FOUNDED",
    "lead": "LEADS",
    "head": "LEADS",
    "run": "LEADS",
    "manage": "LEADS",
    "chair": "LEADS",
    "work": "WORKS_FOR",
    "employ": "WORKS_FOR",
    "locate": "LOCATED_IN",
    "base": "LOCATED_IN",
    "situate": "LOCATED_IN",
    "partner": "PARTNERED_WITH",
    "collaborate": "PARTNERED_WITH",
    "cooperate": "PARTNERED_WITH",
    "acquire": "ACQUIRED",
    "buy": "ACQUIRED",
    "purchase": "ACQUIRED",
    "invest": "INVESTED_IN",
    "fund": "INVESTED_IN",
    "join": "MEMBER_OF",
    "own": "SUBSIDIARY_OF",
}

NER_CONFIDENCE = 0.8
RELATION_CONFIDENCE = 0.6


def _governing_verb(root_a, root_b):
    """Lowest common ancestor token of two entity roots (or None)."""
    ancestors: dict[int, object] = {}
    token = root_a
    while True:
        ancestors[token.i] = token
        if token.head.i == token.i:
            break
        token = token.head
    token = root_b
    while True:
        if token.i in ancestors:
            return ancestors[token.i]
        if token.head.i == token.i:
            break
        token = token.head
    return None


def _is_argument(root, verb) -> bool:
    """True when ``root`` is a direct (or one-preposition-hop) argument of verb."""
    head = root.head
    if head.i == verb.i:
        return True
    return head.pos_ == "ADP" and head.head.i == verb.i


def _is_subject(root) -> bool:
    return root.dep_ in ("nsubj", "nsubjpass")


class SpacyExtractor:
    name = "spacy"

    def __init__(self, model: str = "en_core_web_sm") -> None:
        try:
            import spacy  # lazy: optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "spaCy is not installed. Run: pip install -e .[extraction]"
            ) from exc

        try:
            self._nlp = spacy.load(model)
        except OSError as exc:  # model not downloaded
            raise RuntimeError(
                f"spaCy model {model!r} not found. Run: "
                f"python -m spacy download {model}"
            ) from exc

    def extract(self, text: str, doc_key: str = "") -> ExtractionResult:
        doc = self._nlp(text)

        mentions: list[Mention] = []
        seen_spans: set[tuple[int, int]] = set()
        for ent in doc.ents:
            surface = ent.text.strip()
            if not surface or (ent.start_char, ent.end_char) in seen_spans:
                continue
            seen_spans.add((ent.start_char, ent.end_char))
            mentions.append(
                Mention(
                    surface=surface,
                    entity_type=LABEL_MAP.get(ent.label_, "MISC"),
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=NER_CONFIDENCE,
                )
            )

        relations: list[RelationProposal] = []
        seen_relations: set[tuple[str, str, str]] = set()
        for sent in doc.sents:
            entities = [ent for ent in sent.ents if ent.text.strip()]
            evidence = sent.text.strip()
            for entity_a, entity_b in combinations(entities, 2):
                if entity_a.text.strip().lower() == entity_b.text.strip().lower():
                    continue
                verb = _governing_verb(entity_a.root, entity_b.root)
                if verb is None or verb.pos_ not in ("VERB", "AUX"):
                    continue
                if not (_is_argument(entity_a.root, verb)
                        and _is_argument(entity_b.root, verb)):
                    continue
                a_subject = _is_subject(entity_a.root)
                b_subject = _is_subject(entity_b.root)
                if a_subject == b_subject:
                    continue  # need exactly one subject and one object
                source, target = (entity_b, entity_a) if b_subject else (entity_a, entity_b)
                relation = VERB_RELATIONS.get(verb.lemma_.lower(), "RELATED_TO")
                key = (source.text.strip(), target.text.strip(), relation)
                if key in seen_relations:
                    continue
                seen_relations.add(key)
                relations.append(
                    RelationProposal(
                        source_surface=source.text.strip(),
                        target_surface=target.text.strip(),
                        relation=relation,
                        evidence=evidence,
                        confidence=RELATION_CONFIDENCE,
                    )
                )

        log.info("spaCy extracted %d mentions, %d relations for %s",
                 len(mentions), len(relations), doc_key or "document")
        return ExtractionResult(doc_key=doc_key, mentions=mentions, relations=relations)
