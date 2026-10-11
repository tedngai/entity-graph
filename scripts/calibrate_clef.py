"""Calibrate a SystemOne provider's relation-judgment prompts against the gold set.

The model is fixed; only the ``instructions``/``criteria`` sent in each question
change. Scores the same gate as ``compare_providers.py`` (promote == role).

    python scripts/calibrate_clef.py --url http://192.168.1.242:8001/v1/systemone \
        --model Cloudflare/clef-flash
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from entity_graph.config import load_config  # noqa: E402
from entity_graph.eval_facts import load_gold  # noqa: E402
from entity_graph.extract.gate import RelationJudgment, should_promote  # noqa: E402
from entity_graph.extract.jev import RELATION_VALIDATION_THRESHOLD, SystemOneClient  # noqa: E402


def rel_words(relation: str) -> str:
    return str(relation).lower().replace("_", " ")


# --- assertion-question variants -------------------------------------------

def assertion_baseline(c: dict) -> str:
    return (
        f"For claim `claim_{c['id']}`, probability that its `evidence` asserts "
        f"that its `subject` {rel_words(c['relation'])} its `object`? "
        f"Answer for that claim only."
    )


def assertion_explicit(c: dict) -> str:
    return (
        f"Claim `claim_{c['id']}`: `subject` {rel_words(c['relation'])} `object`. "
        f"Read the evidence. Give the probability that the evidence states THIS "
        f"exact directed claim: the subject is the doer and the object is the "
        f"receiver, and it is stated as done or true. If the evidence reverses "
        f"the two roles, describes a different relation, or only expresses a "
        f"plan, intention, possibility, or denial, give a LOW probability. "
        f"Answer for that claim only."
    )


def assertion_explicit_short(c: dict) -> str:
    return (
        f"For claim `claim_{c['id']}`, probability that its `evidence` asserts "
        f"exactly `subject` {rel_words(c['relation'])} `object` (same direction, "
        f"not the reverse, not a different relation) as something done or true, "
        f"not merely planned or possible? Answer for that claim only."
    )


def assertion_passive(c: dict) -> str:
    return (
        f"For claim `claim_{c['id']}`, probability that its `evidence` asserts exactly "
        f"`subject` {rel_words(c['relation'])} `object`, in that direction, as something "
        f"done or true. Mind passive voice: in phrases like 'founded by X' or 'led by X', "
        f"X is the doer, so the claim's subject must be X. If the evidence reverses the "
        f"roles, states a different relation, or only plans or predicts it, give a LOW "
        f"probability. Answer for that claim only."
    )


def assertion_doer(c: dict) -> str:
    return (
        f"For claim `claim_{c['id']}`, probability that its `evidence` asserts that "
        f"`{c['subject']}` is the doer and `{c['object']}` is the receiver of "
        f"'{rel_words(c['relation'])}', stated as something done or true. If the evidence "
        f"reverses the roles (for example passive voice such as 'founded by X', where X is "
        f"the doer), states a different relation, or only plans or predicts it, give a LOW "
        f"probability. Answer for that claim only."
    )


# --- modality-criteria variants --------------------------------------------

MODALITY_BASELINE = {
    "fact": "The evidence states this as something that happened or is true.",
    "possibility": "The evidence only says this might happen or could be true.",
    "denial": "The evidence says this did not happen or will not happen.",
    "forecast": "The evidence says this is planned or expected, not yet done.",
    "not_stated": "The evidence does not state this claim.",
}

MODALITY_EXPLICIT = {
    "fact": "The evidence states the claim as already true or completed: past or "
            "present tense such as did, has, is, was, announced.",
    "possibility": "The evidence says it might or could happen, with no commitment.",
    "denial": "The evidence says it did not happen or will not happen.",
    "forecast": "The evidence says it will happen, plans to, intends to, is expected "
                "to, or is to be done later: future or planned, not yet done.",
    "not_stated": "The evidence does not state this exact directed claim: the roles "
                  "are reversed, it is a different relation, or it is absent.",
}

MODALITY_STRICT = {
    "fact": "A completed or currently-true statement: did, has, is, was, announced, "
            "founded, led by. Past or present tense.",
    "possibility": "A maybe: might, could, may. No commitment.",
    "denial": "A negative: did not, will not, never, denies.",
    "forecast": "A future or planned statement: will, plans to, intends to, expects "
                "to, is to, aims to, is going to. Not yet done.",
    "not_stated": "Wrong direction (roles reversed), a different relation, or not in "
                  "the evidence at all.",
}

MODALITY_TENSE = {
    "fact": "Past or present tense, already true: is, was, has, did, announced, "
            "founded, led by, works, employs.",
    "possibility": "might, could, may, perhaps. No commitment.",
    "denial": "did not, will not, never, denies, refutes.",
    "forecast": "Future or intended: will, plans to, intends to, expects to, aims to, "
                "is to, is going to, expansion into, plans to build, is set to. "
                "The action is NOT done yet.",
    "not_stated": "Roles reversed, a different relation, or absent from the evidence.",
}

# Directs attention to future cue words, since the model tends to read the main
# verb ("cooperate") as present and miss the modal ("will").
MODALITY_FUTURE = {
    "fact": "Already done or currently true (past or present): is, was, has, did, "
            "announced, founded, led by, works.",
    "possibility": "might, could, may, perhaps. No commitment.",
    "denial": "did not, will not, never, denies, refutes.",
    "forecast": "FUTURE or PLANNED. Choose this whenever the evidence contains will, "
                "plans to, intends to, expects to, aims to, is to, is going to, "
                "expansion into, or plans to build -- even if the action sounds "
                "cooperative or positive. The action is NOT done yet.",
    "not_stated": "Roles reversed, a different relation, or absent from the evidence.",
}

# A simpler 3-way modality question (mapped back to gate labels).
MODALITY_BINARY = {
    "completed_or_true": "The evidence states the claim as already done or currently "
                         "true: is, was, has, did, announced, founded, led by.",
    "planned_or_future": "The evidence states the claim as future or intended, NOT yet "
                         "done: will, plans to, intends to, expects to, aims to, is to, "
                         "is going to, expansion into, plans to build.",
    "neither": "The evidence denies it, only says it might happen, reverses the two "
               "roles, states a different relation, or does not state it.",
}
MODALITY_BINARY_MAP = {
    "completed_or_true": "fact",
    "planned_or_future": "forecast",
    "neither": "not_stated",
}

DIRECTION_CRITERIA = {
    "subject_to_object": "The evidence states the subject acts on the object: the "
                         "subject is the doer and the object receives the action.",
    "object_to_subject": "The evidence states the object acts on the subject: the two "
                         "roles are reversed relative to the claim.",
    "not_stated": "The evidence does not state this relation between the two in either "
                  "direction.",
}

# --- state-task variants ---------------------------------------------------

TASK_BASELINE = (
    "For each claim, decide whether its evidence asserts the relation, and what "
    "kind of statement that evidence is."
)

TASK_EXPLICIT = (
    "For each claim, decide (a) whether its evidence asserts that exact directed "
    "claim -- subject doing it to object, stated as done or true -- and (b) what "
    "kind of statement the evidence is (completed fact, plan/forecast, "
    "possibility, denial, or not stated)."
)

# Few-shot examples, drawn from the failure modes (reversal, forecast, correct).
EXAMPLES = [
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
]

# Adds passive-voice reversal and "will" plans to the examples.
EXAMPLES2 = EXAMPLES + [
    {"claim": {"subject": "Acme", "relation": "FOUNDED", "object": "Alice"},
     "evidence": "founded by Alice in 2019.",
     "assertion": 0.02, "modality": "not_stated"},
    {"claim": {"subject": "Acme", "relation": "PARTNERED_WITH", "object": "Beta"},
     "evidence": "Acme will cooperate with Beta.",
     "assertion": 0.90, "modality": "forecast"},
    {"claim": {"subject": "Acme", "relation": "LOCATED_IN", "object": "Berlin"},
     "evidence": "Berlin-based Acme opened offices in Berlin.",
     "assertion": 0.95, "modality": "fact"},
]

VARIANTS: dict[str, dict] = {
    "baseline": {"assertion": assertion_baseline, "modality": MODALITY_BASELINE,
                 "task": TASK_BASELINE, "examples": False, "split": False, "direction": False},
    "explicit": {"assertion": assertion_explicit, "modality": MODALITY_EXPLICIT,
                 "task": TASK_EXPLICIT, "examples": False, "split": False, "direction": False},
    "explicit_short": {"assertion": assertion_explicit_short, "modality": MODALITY_EXPLICIT,
                       "task": TASK_EXPLICIT, "examples": False, "split": False, "direction": False},
    "strict": {"assertion": assertion_explicit, "modality": MODALITY_STRICT,
               "task": TASK_EXPLICIT, "examples": False, "split": False, "direction": False},
    "fewshot": {"assertion": assertion_explicit, "modality": MODALITY_STRICT,
                "task": TASK_EXPLICIT, "examples": True, "split": False, "direction": False},
    "fewshot_short": {"assertion": assertion_explicit_short, "modality": MODALITY_STRICT,
                      "task": TASK_EXPLICIT, "examples": True, "split": False, "direction": False},
    "tense": {"assertion": assertion_explicit_short, "modality": MODALITY_TENSE,
              "task": TASK_EXPLICIT, "examples": True, "split": False, "direction": False},
    "tense_split": {"assertion": assertion_explicit_short, "modality": MODALITY_TENSE,
                    "task": TASK_EXPLICIT, "examples": True, "split": True, "direction": False},
    "tense_split_dir": {"assertion": assertion_explicit_short, "modality": MODALITY_TENSE,
                        "task": TASK_EXPLICIT, "examples": True, "split": True, "direction": True},
    "binary": {"assertion": assertion_explicit_short, "modality": MODALITY_BINARY,
               "task": TASK_EXPLICIT, "examples": EXAMPLES2, "split": False, "direction": False,
               "modality_map": MODALITY_BINARY_MAP},
    "binary_split": {"assertion": assertion_explicit_short, "modality": MODALITY_BINARY,
                     "task": TASK_EXPLICIT, "examples": EXAMPLES2, "split": True, "direction": False,
                     "modality_map": MODALITY_BINARY_MAP},
    "future": {"assertion": assertion_explicit_short, "modality": MODALITY_FUTURE,
               "task": TASK_EXPLICIT, "examples": EXAMPLES2, "split": False, "direction": False},
    "targeted": {"assertion": assertion_passive, "modality": MODALITY_FUTURE,
                 "task": TASK_EXPLICIT, "examples": EXAMPLES2, "split": False, "direction": False},
    "doer": {"assertion": assertion_doer, "modality": MODALITY_TENSE,
             "task": TASK_EXPLICIT, "examples": EXAMPLES2, "split": False, "direction": False},
}


def _answers(response: object) -> dict:
    if isinstance(response, dict):
        answers = response.get("answers")
        if isinstance(answers, dict):
            return answers
    return {}


def run_variant(client: SystemOneClient, gold, variant: dict, batch_size: int = 12) -> dict:
    claims = [
        {"id": i, "subject": f.subject, "relation": f.relation, "object": f.object,
         "evidence": f.evidence}
        for i, f in enumerate(gold)
    ]
    assertion: dict[int, float] = {}
    modality: dict[int, str] = {}
    direction: dict[int, str] = {}
    started = time.perf_counter()
    for start in range(0, len(claims), batch_size):
        batch = claims[start:start + batch_size]

        def make_questions(with_modality: bool) -> dict:
            questions: dict = {}
            for c in batch:
                questions[f"claim_{c['id']}"] = {
                    "type": "noul", "instructions": variant["assertion"](c)}
                if variant["direction"]:
                    questions[f"direction_{c['id']}"] = {
                        "type": "choice",
                        "instructions": f"For claim `claim_{c['id']}`, does the evidence state "
                                        f"`subject` {rel_words(c['relation'])} `object` (subject as "
                                        f"doer), the reverse, or neither?",
                        "criteria": DIRECTION_CRITERIA,
                    }
                if with_modality:
                    questions[f"modality_{c['id']}"] = {
                        "type": "choice",
                        "instructions": f"For claim `claim_{c['id']}`, what kind of statement "
                                        f"is its evidence about that claim?",
                        "criteria": variant["modality"],
                    }
            return questions

        def state() -> dict:
            s: dict = {"task": variant["task"], "claims": batch}
            examples = variant["examples"]
            if examples:
                s["examples"] = examples if isinstance(examples, list) else EXAMPLES
            return s

        requests = [make_questions(not variant["split"])]
        if variant["split"]:
            requests.append(make_questions(True))
        responses = []
        for qs in requests:
            try:
                responses.append(client.decide(state(), qs))
            except Exception as exc:  # noqa: BLE001
                print(f"  call failed: {exc}")
                responses.append({})

        for response in responses:
            answers = _answers(response)
            for c in batch:
                cid = c["id"]
                a = answers.get(f"claim_{cid}", {})
                m = answers.get(f"modality_{cid}", {})
                d = answers.get(f"direction_{cid}", {})
                if isinstance(a, dict) and a.get("noul") is not None:
                    assertion[cid] = float(a["noul"])
                if isinstance(m, dict) and m.get("choice") is not None:
                    modality[cid] = str(m["choice"])
                if isinstance(d, dict) and d.get("choice") is not None:
                    direction[cid] = str(d["choice"])
    elapsed = time.perf_counter() - started

    correct = leaks = missed = 0
    details = []
    for i, fact in enumerate(gold):
        raw_modality = modality.get(i)
        gate_modality = variant.get("modality_map", {}).get(raw_modality, raw_modality)
        judgment = RelationJudgment(
            assertion=assertion.get(i), modality=gate_modality, source="jev",
        )
        promoted = should_promote(judgment, threshold=RELATION_VALIDATION_THRESHOLD)
        # A provider-specific direction check: reversed or absent -> not a fact.
        if variant["direction"] and direction.get(i) not in (None, "subject_to_object"):
            promoted = False
        expected = fact.role == "fact"
        ok = promoted == expected
        correct += ok
        if promoted and not expected:
            leaks += 1
            details.append(("LEAK", fact))
        if not promoted and expected:
            missed += 1
            details.append(("MISS", fact))
    return {"acc": correct / len(gold), "leaks": leaks, "missed": missed,
            "elapsed": elapsed, "details": details,
            "verdicts": {i: (assertion.get(i), modality.get(i), direction.get(i))
                         for i in range(len(gold))}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", default="data/gold/greengrid_partnership.jsonl")
    parser.add_argument("--url", default="http://192.168.1.242:8001/v1/systemone")
    parser.add_argument("--model", default="Cloudflare/clef-flash")
    parser.add_argument("--key", default="local")
    parser.add_argument("--variant", default=None, help="run only this variant")
    parser.add_argument("--batch-size", type=int, default=12,
                        help="claims per request (context load; try 1, 2, 4, 12)")
    parser.add_argument("--dump", action="store_true",
                        help="print every claim's score, not just errors")
    args = parser.parse_args(argv)

    gold = load_gold(args.gold)
    cfg = replace(load_config(), systemone_url=args.url, systemone_model=args.model,
                  systemone_api_key=args.key)
    client = SystemOneClient(cfg)

    names = [args.variant] if args.variant else list(VARIANTS)
    print(f"batch_size={args.batch_size}")
    print(f"{'variant':16} {'acc':>6} {'leaks':>6} {'missed':>7} {'time':>6}")
    print("-" * 46)
    results = {}
    for name in names:
        result = run_variant(client, gold, VARIANTS[name], batch_size=args.batch_size)
        results[name] = result
        print(f"{name:16} {result['acc']:>6.3f} {result['leaks']:>6} {result['missed']:>7} "
              f"{result['elapsed']:>5.2f}s")
        for tag, fact in result["details"]:
            i = gold.index(fact)
            noul, mod, direction = result["verdicts"].get(i, (None, None, None))
            print(f"    {tag}: {fact.role:8} {fact.subject[:24]:24} {fact.relation:14} "
                  f"{fact.object[:20]:20} noul={noul} mod={mod} dir={direction}")
        if args.dump:
            for i, fact in enumerate(gold):
                noul, mod, direction = result["verdicts"].get(i, (None, None, None))
                noul_s = f"{noul:.3f}" if isinstance(noul, float) else str(noul)
                print(f"    [{i:2}] {fact.role:8} {fact.subject[:22]:22} "
                      f"{fact.relation:14} {fact.object[:18]:18} noul={noul_s:>6} mod={mod}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
