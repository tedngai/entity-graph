"""Provider prompt profiles for relation judgment.

Jev and compatible System One endpoints are asked the same *questions*, but they
can respond very differently to the *wording*. A profile bundles the task text,
the assertion instruction, the modality criteria, optional few-shot examples,
and the request batch size, so an endpoint can be calibrated without forking the
client.

The ``default`` profile is the wording live-verified against Jev (see
``docs/RESEARCH.md`` §4b) and is used unless a profile is selected. The ``clef``
profile is the calibrated wording for a small local model (§4d): it names the
doer/receiver and the tense cues, adds few-shot examples, and keeps each request
to one claim, because a small model's answers degrade as more claims share a
context. Selecting a profile is opt-in via ``SYSTEMONE_PROFILE`` or the
``profile=`` argument; it never changes the frozen gate threshold (D15).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


def rel_words(relation: str) -> str:
    """``WORKS_FOR`` -> ``works for`` (the phrasing the models were verified on)."""
    return str(relation).lower().replace("_", " ")


# --- assertion questions ----------------------------------------------------

def assertion_plain(claim: dict) -> str:
    """Default wording: a bare probability that the evidence asserts the claim."""
    return (
        f"For claim `claim_{claim['id']}`, probability that its `evidence` "
        f"asserts that its `subject` {rel_words(claim['relation'])} its "
        f"`object`? Answer for that claim only."
    )


def assertion_doer(claim: dict) -> str:
    """Calibrated wording: name the doer/receiver and call out role reversal.

    A small model reads passive voice (``founded by X``) as if the grammatical
    subject were the doer. Naming the doer and the receiver, and warning that
    passive voice flips them, recovers the reversal without lowering true
    positives (see ``docs/RESEARCH.md`` §4d).
    """
    return (
        f"For claim `claim_{claim['id']}`, probability that its `evidence` "
        f"asserts that `{claim['subject']}` is the doer and `{claim['object']}` "
        f"is the receiver of '{rel_words(claim['relation'])}', stated as "
        f"something done or true. If the evidence reverses the roles (for "
        f"example passive voice such as 'founded by X', where X is the doer), "
        f"states a different relation, or only plans or predicts it, give a LOW "
        f"probability. Answer for that claim only."
    )


# --- modality criteria ------------------------------------------------------

# Default: short labels verified against Jev.
MODALITY_CRITERIA = {
    "fact": "The evidence states this as something that happened or is true.",
    "possibility": "The evidence only says this might happen or could be true.",
    "denial": "The evidence says this did not happen or will not happen.",
    "forecast": "The evidence says this is planned or expected, not yet done.",
    "not_stated": "The evidence does not state this claim.",
}

# Calibrated: spell out the tense/lexical cues. Without the cues a small model
# reads the main verb ("cooperate") as present and misses the modal ("will"),
# promoting forecasts to facts.
MODALITY_TENSE = {
    "fact": "Past or present tense, already true: is, was, has, did, announced, "
            "founded, led by, works, employs.",
    "possibility": "might, could, may, perhaps. No commitment.",
    "denial": "did not, will not, never, denies, refutes.",
    "forecast": "Future or intended: will, plans to, intends to, expects to, aims "
                "to, is to, is going to, expansion into, plans to build, is set to. "
                "The action is NOT done yet.",
    "not_stated": "Roles reversed, a different relation, or absent from the evidence.",
}


# --- few-shot examples (calibrated profile only) ----------------------------

# Drawn from the failure modes: role reversal, passive-voice reversal, a plan,
# and a correctly asserted fact.
EXAMPLES = (
    {"claim": {"subject": "Acme", "relation": "ACQUIRED", "object": "Beta"},
     "evidence": "Acme acquired Beta in 2020.",
     "assertion": 0.97, "modality": "fact"},
    {"claim": {"subject": "Acme", "relation": "ACQUIRED", "object": "Beta"},
     "evidence": "Beta was acquired by Acme.",
     "assertion": 0.95, "modality": "fact"},
    {"claim": {"subject": "Beta", "relation": "ACQUIRED", "object": "Acme"},
     "evidence": "Acme acquired Beta in 2020.",
     "assertion": 0.02, "modality": "not_stated"},
    {"claim": {"subject": "Acme", "relation": "INVESTED_IN", "object": "Beta"},
     "evidence": "Acme will invest 400 million euros in Beta.",
     "assertion": 0.85, "modality": "forecast"},
    {"claim": {"subject": "Acme", "relation": "SUBSIDIARY_OF", "object": "Beta"},
     "evidence": "Acme announced a strategic partnership with Beta.",
     "assertion": 0.02, "modality": "not_stated"},
    {"claim": {"subject": "Acme", "relation": "FOUNDED", "object": "Alice"},
     "evidence": "founded by Alice in 2019.",
     "assertion": 0.02, "modality": "not_stated"},
    {"claim": {"subject": "Acme", "relation": "PARTNERED_WITH", "object": "Beta"},
     "evidence": "Acme will cooperate with Beta.",
     "assertion": 0.90, "modality": "forecast"},
    {"claim": {"subject": "Acme", "relation": "LOCATED_IN", "object": "Berlin"},
     "evidence": "Berlin-based Acme opened offices in Berlin.",
     "assertion": 0.95, "modality": "fact"},
)


@dataclass(frozen=True)
class RelationPromptProfile:
    """Wording and request shape for one provider's relation judgment."""

    name: str
    task: str
    assertion: Callable[[dict], str]
    modality_criteria: dict[str, str]
    batch_size: int = 12
    examples: tuple[dict, ...] = ()
    # Provider label -> gate label (see gate.normalize_modality). Empty when the
    # provider already answers with the gate's own labels.
    modality_map: dict[str, str] = field(default_factory=dict)


DEFAULT_PROFILE = RelationPromptProfile(
    name="default",
    task=(
        "For each claim, decide whether its evidence asserts the relation, and "
        "what kind of statement that evidence is."
    ),
    assertion=assertion_plain,
    modality_criteria=MODALITY_CRITERIA,
    batch_size=12,
)

CLEF_PROFILE = RelationPromptProfile(
    name="clef",
    task=(
        "For each claim, decide (a) whether its evidence asserts that exact "
        "directed claim -- subject doing it to object, stated as done or true -- "
        "and (b) what kind of statement the evidence is (completed fact, "
        "plan/forecast, possibility, denial, or not stated)."
    ),
    assertion=assertion_doer,
    modality_criteria=MODALITY_TENSE,
    batch_size=1,
    examples=EXAMPLES,
)

PROFILES: dict[str, RelationPromptProfile] = {
    DEFAULT_PROFILE.name: DEFAULT_PROFILE,
    CLEF_PROFILE.name: CLEF_PROFILE,
}


def get_profile(name: str | None) -> RelationPromptProfile:
    """Look up a profile by name; empty/None selects the default."""
    if not name:
        return DEFAULT_PROFILE
    try:
        return PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown SYSTEMONE_PROFILE {name!r}; known: {known}") from None
