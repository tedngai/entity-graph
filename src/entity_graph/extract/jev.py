"""Jev (System One) client — typed, calibrated decisions (TypeSafe).

Jev is NOT an NER model: it does not find spans. It answers questions we ask
about candidate spans/entities with typed answers and calibrated probabilities:

    Choice -> one of N options          (e.g. entity type arbitration)
    Score  -> ordinal level w/ labeled criteria (e.g. 3-way entity alignment)
    Noul   -> probability of true       (e.g. "does this sentence assert X?")

Docs: https://docs.typesafe.ai/concepts/system-one
API reference: https://docs.typesafe.ai/api
Entity-alignment pattern: https://docs.typesafe.ai/cookbooks/entity_alignment

Contract (verified against the published API reference, 2026-10-10):
  request  {model, state, questions}
  question {type: noul|choice|score, instructions, criteria?}
           - all types use ``instructions`` (string|object|array)
           - choice ``criteria`` is a MAP option -> description (2..255)
           - score  ``criteria`` is an ORDERED ARRAY of 2..10 levels
           - noul   ``criteria`` is optional {true, false}
  response {model, answers: {id: answer}, usage}
           - noul   answer: {type, noul}         (no confidence)
           - choice answer: {type, choice, probabilities, confidence}
           - score  answer: {type, score, legend, probabilities, confidence}
  Endpoint: POST https://api.typesafe.ai/v1/systemone (configurable). OpenRouter
  exposes the same body at POST https://openrouter.ai/api/alpha/decisions with
  model ``typesafe/jev-1.13``. The Pratt router exposes it at
  POST https://llm.pratt.edu/v1/systemone with model ``pratt-jev``.

When none of SYSTEMONE_API_KEY / PRATT_LLM_API_KEY / OPENROUTER_API_KEY is set,
every method returns None (the pipeline continues on exact/alias/trigram
resolution), so the project runs fully offline. Live-verified 2026-10-11 (P5.6)
via the OpenRouter mirror — see docs/RESEARCH.md §4b — and via the Pratt router
(`/v1/systemone`, model ``pratt-jev``).
"""

from __future__ import annotations

import http.client
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from ..config import Config, load_config
from .gate import RelationJudgment, normalize_modality

log = logging.getLogger(__name__)


class SystemOneError(RuntimeError):
    pass


# The three labeled criteria levels of the alignment Score question,
# straight from the official entity-alignment cookbook. Ordering matters:
# the outcome is chosen by rounding the score to the nearest level, so there
# are no threshold constants in our code.
ALIGNMENT_CRITERIA = [
    "different entities: names, kind and context disagree -> leave unlinked",
    "related but not certain: shared words or domain, conflicting fields "
    "-> send to curator queue",
    "same entity: name, kind and context agree -> assert merge (sameAs)",
]

# FROZEN. A relation proposal whose Jev Noul ("does the evidence assert this?")
# falls below this value is rejected rather than promoted. Proposals Jev could
# not score (disabled / error) are promoted on the extractor's own confidence.
# Do not tune this from a single sample. `entity-graph eval` reports a sweep
# against data/gold/; changing the constant requires a new decision record (D15).
RELATION_VALIDATION_THRESHOLD = 0.5

# Short Choice labels for the modality question asked beside the assertion Noul.
# Argmax only — the promotion rule does not threshold these probabilities.
MODALITY_CRITERIA = {
    "fact": "The evidence states this as something that happened or is true.",
    "possibility": "The evidence only says this might happen or could be true.",
    "denial": "The evidence says this did not happen or will not happen.",
    "forecast": "The evidence says this is planned or expected, not yet done.",
    "not_stated": "The evidence does not state this claim.",
}


@dataclass
class AlignmentVerdict:
    outcome: str  # "separate" | "curator" | "merge"
    score: float
    sidecars: dict[str, float] = field(default_factory=dict)


class SystemOneClient:
    """Minimal HTTP client for POST /v1/systemone (stdlib only)."""

    def __init__(self, config: Config | None = None, timeout: float = 20.0,
                 retries: int = 2) -> None:
        cfg = config or load_config()
        self.url = cfg.systemone_url
        self.api_key = cfg.systemone_api_key
        self.model = cfg.systemone_model
        self.timeout = timeout
        # Transient network failures (a dropped connection, a 5xx) are retried
        # this many times before surfacing as SystemOneError, which callers
        # treat as "unavailable" and degrade past.
        self.retries = retries
        # OpenRouter attribution headers (harmless for the native endpoint).
        self.headers = {}
        if cfg.openrouter_referer:
            self.headers["HTTP-Referer"] = cfg.openrouter_referer
        if cfg.openrouter_title:
            self.headers["X-OpenRouter-Title"] = cfg.openrouter_title

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def decide(self, state: Any, questions: dict[str, dict]) -> dict:
        """One call, many questions. `state` is text/JSON (keep it small).

        `questions` maps an id to a primitive config, e.g.::

            {
              "same_entity": {"type": "score",
                              "instructions": "...",
                              "criteria": ALIGNMENT_CRITERIA},
              "name_match":  {"type": "noul",
                              "instructions": "Probability that ..."},
              "entity_type": {"type": "choice",
                              "instructions": "...",
                              "criteria": {"PERSON": "...", "ORG": "..."}},
            }

        Returns the raw API JSON: ``{"model", "answers": {id: {...}}, "usage"}``.
        """
        if not self.enabled:
            raise SystemOneError("SYSTEMONE_API_KEY is not set; Jev decisions disabled.")
        payload = {"model": self.model, "state": state, "questions": questions}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            **self.headers,
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        attempt = 0
        while True:
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:800]
                except Exception:  # noqa: BLE001 - best-effort diagnostics
                    pass
                # A 5xx is worth retrying; a 4xx is not (bad key, bad request).
                if exc.code >= 500 and attempt < self.retries:
                    attempt += 1
                    time.sleep(min(2.0, 0.5 * attempt))
                    continue
                raise SystemOneError(
                    f"SystemOne call failed: HTTP {exc.code} {exc.reason}. {detail}"
                ) from exc
            except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
                # OSError covers URLError and http.client.RemoteDisconnected
                # (a subclass of ConnectionResetError); HTTPException covers
                # BadStatusLine. A dropped connection must not abort the run.
                if attempt < self.retries:
                    attempt += 1
                    time.sleep(min(2.0, 0.5 * attempt))
                    continue
                raise SystemOneError(f"SystemOne call failed: {exc}") from exc


def _get_answer(response: Any, question_id: str) -> Any:
    """Return the answer object for `question_id`, or None.

    Primary shape is ``{"answers": {id: answer}}``. A bare ``{id: answer}`` and
    a scalar value are tolerated for forward compatibility.
    """
    if isinstance(response, dict):
        answers = response.get("answers")
        if isinstance(answers, dict) and question_id in answers:
            return answers[question_id]
        if question_id in response:
            return response[question_id]
    return None


def _numeric_answer(response: Any, question_id: str, field: str) -> float | None:
    """Extract a float from ``answers[id][field]`` (e.g. noul/score).

    Also accepts a bare numeric answer, so a simplified/older response shape
    still parses. Returns None when the field is absent or non-numeric.
    """
    answer = _get_answer(response, question_id)
    if isinstance(answer, dict):
        value = answer.get(field)
    elif isinstance(answer, (int, float)):
        value = answer
    else:
        value = None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def align_pair(client: SystemOneClient, entity_a: dict, entity_b: dict) -> AlignmentVerdict | None:
    """Entity alignment per the official cookbook.

    Both entity profiles go into ONE state as entity_a/entity_b. One Score
    question with the three labeled criteria routes the pair; Noul sidecars
    per field (name/kind/context) ride in the same request so a curator can
    see which fields disagree. Route = round(score) -> outcome. No thresholds.
    """
    if not client.enabled:
        return None
    state = {
        "task": "Decide whether two candidate knowledge-graph entities are the "
                "same real-world thing.",
        "entity_a": entity_a,
        "entity_b": entity_b,
    }
    questions: dict[str, dict] = {
        "same_entity": {
            "type": "score",
            "instructions": "How do `entity_a` and `entity_b` relate as "
                            "real-world things? Pick the level that matches.",
            "criteria": ALIGNMENT_CRITERIA,
        },
    }
    for field_name in ("name", "kind", "context"):
        questions[f"{field_name}_match"] = {
            "type": "noul",
            "instructions": f"Probability that the {field_name} fields of "
                            f"`entity_a` and `entity_b` refer to the same thing?",
        }
    try:
        response = client.decide(state, questions)
    except SystemOneError as exc:
        log.warning("Jev alignment unavailable (%s); keeping pair unresolved.", exc)
        return None

    score = _numeric_answer(response, "same_entity", "score")
    if score is None:
        log.warning("Jev response missing 'same_entity' score; response: %r", response)
        return None
    outcome = ("separate", "curator", "merge")[max(0, min(2, round(score)))]
    sidecars = {}
    for field_name in ("name", "kind", "context"):
        value = _numeric_answer(response, f"{field_name}_match", "noul")
        if value is not None:
            sidecars[field_name] = value
    return AlignmentVerdict(outcome=outcome, score=score, sidecars=sidecars)


def validate_relations(
    client: SystemOneClient, claims: list[dict], batch_size: int = 25
) -> dict[int, float]:
    """Noul-validate many relation claims in as few calls as possible.

    ``claims`` items: ``{"id": int, "subject": str, "relation": str,
    "object": str, "evidence": str}``. Returns ``{id: noul}`` for the claims
    the API answered. One request carries up to ``batch_size`` questions
    (state + per-claim question ids), which keeps cost/latency low.
    """
    if not client.enabled or not claims:
        return {}
    verdicts: dict[int, float] = {}
    for start in range(0, len(claims), batch_size):
        batch = claims[start : start + batch_size]
        state = {
            "task": "For each claim, decide whether its evidence sentence asserts "
                    "that its subject stands in the given relation to its object.",
            "claims": batch,
        }
        questions = {
            f"claim_{claim['id']}": {
                "type": "noul",
                "instructions": f"For claim `claim_{claim['id']}`, probability that "
                                f"its `evidence` asserts that its `subject` "
                                f"{str(claim['relation']).lower().replace('_', ' ')} "
                                f"its `object`? Answer for that claim only.",
            }
            for claim in batch
        }
        try:
            response = client.decide(state, questions)
        except SystemOneError as exc:
            log.warning("Jev relation validation unavailable (%s)", exc)
            continue
        for claim in batch:
            value = _numeric_answer(response, f"claim_{claim['id']}", "noul")
            if value is not None:
                verdicts[claim["id"]] = value
    return verdicts


def _choice_answer(response: Any, question_id: str) -> str | None:
    """Extract ``answers[id].choice``. A bare string is tolerated."""
    answer = _get_answer(response, question_id)
    if isinstance(answer, dict):
        value = answer.get("choice")
    elif isinstance(answer, str):
        value = answer
    else:
        value = None
    text = str(value).strip() if value is not None else ""
    return text or None


def judge_relations(
    client: SystemOneClient, claims: list[dict], batch_size: int = 12
) -> dict[int, RelationJudgment]:
    """Noul-validate claims and ask a modality Choice in the same request.

    ``claims`` items match ``validate_relations``. Modality is a separate
    question id so it sits beside the assertion Noul rather than replacing it.
    A claim the API does not answer is omitted (treated as unscored).
    """
    if not client.enabled or not claims:
        return {}
    verdicts: dict[int, RelationJudgment] = {}
    for start in range(0, len(claims), batch_size):
        batch = claims[start : start + batch_size]
        state = {
            "task": "For each claim, decide whether its evidence asserts the "
                    "relation, and what kind of statement that evidence is.",
            "claims": batch,
        }
        questions: dict[str, dict] = {}
        for claim in batch:
            claim_id = claim["id"]
            questions[f"claim_{claim_id}"] = {
                "type": "noul",
                "instructions": (
                    f"For claim `claim_{claim_id}`, probability that its "
                    f"`evidence` asserts that its `subject` "
                    f"{str(claim['relation']).lower().replace('_', ' ')} "
                    f"its `object`? Answer for that claim only."
                ),
            }
            questions[f"modality_{claim_id}"] = {
                "type": "choice",
                "instructions": (
                    f"For claim `claim_{claim_id}`, what kind of statement is "
                    f"its evidence about that claim?"
                ),
                "criteria": MODALITY_CRITERIA,
            }
        try:
            response = client.decide(state, questions)
        except SystemOneError as exc:
            log.warning("Jev relation judgment unavailable (%s)", exc)
            continue
        for claim in batch:
            claim_id = int(claim["id"])
            assertion = _numeric_answer(response, f"claim_{claim_id}", "noul")
            modality = normalize_modality(
                _choice_answer(response, f"modality_{claim_id}")
            )
            if assertion is None and modality is None:
                continue
            verdicts[claim_id] = RelationJudgment(
                assertion=assertion,
                modality=modality,
                agrees=None,
                source="jev",
            )
    return verdicts


def validates_relation(
    client: SystemOneClient, evidence: str, subject: str, relation: str, obj: str
) -> float | None:
    """Noul: probability that the evidence sentence asserts (subject, relation, obj)."""
    if not client.enabled:
        return None
    verdicts = validate_relations(
        client,
        [{"id": 0, "subject": subject, "relation": relation, "object": obj,
          "evidence": evidence}],
    )
    return verdicts.get(0)
