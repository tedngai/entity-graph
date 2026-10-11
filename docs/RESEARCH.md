# RESEARCH.md — Research Findings

Research date: 2026-10-04. Sources were gathered via web search and
developer-indexed searches (GitHub, docs, PyPI, academic indexes). URLs inline.

Questions this document answers:
1. How should entities/relationships/facts be stored in PostgreSQL for graph work?
2. What is the most effective method to extract entities (and relations) from unstructured documents?
3. What is **Jev (SystemOne)** and does it help entity extraction? *(user-specified tool to evaluate)*
4. How do Postgres and NetworkX work together, and how should we analyze/visualize?

---

## 1. Verdict summary

| Question | Verdict |
|---|---|
| Graph storage in Postgres | Relational **edge tables + recursive CTEs** for v1; `pg_trgm` for fuzzy matching, `pgvector` optional for embeddings. Apache AGE only if Cypher-style traversal becomes a real need ([D1](DECISIONS.md#d1)) |
| Entity extraction | **GLiNER2** as primary (zero-shot, schema-conditioned NER + classification + relation extraction in one forward pass, runs locally); **spaCy** for tokenization/sentence-splits and fallback; **LLM only for escalation** on ambiguous documents ([D2](DECISIONS.md#d2)) |
| Jev (SystemOne) | **Yes — use it, but for decisions, not span detection.** Jev is a "System One" decision model returning typed, calibrated answers (`Choice`/`Score`/`Noul`). It is the ideal cheap+calibrated layer for entity alignment, type arbitration, and relation validation. Its official cookbook is literally *knowledge-graph entity alignment* ([D3](DECISIONS.md#d3)). Re-confirmed 2026-10-10 for Jev 1.13 — it still **cannot produce spans**, so it cannot replace the extractor ([§4b](#4b-assessment-2026-10-10-can-jev-replace-gliner2-as-the-extractor), [D7](DECISIONS.md#d7)) |
| Postgres ↔ NetworkX | Load `v_graph_edges` via pandas → `nx.MultiDiGraph` (`nx.from_pandas_edgelist`); analytics in NetworkX, Postgres remains source of truth ([D5](DECISIONS.md#d5)) |
| Visualization | pyvis (interactive HTML) for exploration; matplotlib for static report figures |

---

## 2. Storage: graph data in PostgreSQL

### Options surveyed
- **Plain relational edge tables + `WITH RECURSIVE` CTEs** — the default, portable
  pattern. Recursive CTEs handle bounded traversals; everything else (SQL, joins,
  JSONB attributes, constraints, backups) is standard Postgres.
  Sources: [PuppyGraph — PostgreSQL Graph Database](https://www.puppygraph.com/blog/postgresql-graph-database) (recommends extensions *and* recursive CTEs as the two core patterns).
- **Apache AGE** ("A Graph Extension") — adds openCypher (`MATCH` patterns) inside
  Postgres; graph, relational, and JSON share one engine.
  Sources: [apache/age](https://github.com/apache/age), [PuppyGraph — Apache AGE vs Neo4j](https://www.puppygraph.com/learn/apache-age-vs-neo4j).
  Cost: extra extension to install/operate; not needed while NetworkX does traversal.
- **NewerPostgres-native graph layers** — e.g. `pg_ripple` (RDF/SPARQL + Datalog,
  Postgres 18) and `post-graph`/`pggraph`. Interesting, but heavier or younger than
  needed for v1; revisit if RDF semantics or derived-fact reasoning become goals.
  Sources: [pg-ripple](https://github.com/trickle-labs/pg-ripple), [post-graph](https://pypi.org/project/post-graph/), [evokoa/pggraph](https://github.com/evokoa/pggraph).
- **Dedicated graph DB (Neo4j etc.)** — rejected for v1: the user's stack decision
  is Postgres, and NetworkX covers analytics without a second datastore.

### Schema conclusions (implemented in `db/schema.sql`)
- **Reify facts**: store `(subject, relation, object)` *claims* with evidence quote,
  document provenance, extractor, confidence, and a `valid_from`/`valid_to` window.
  Aggregate edges live in `relationships`; never store evidence only on edges.
- **Keep mentions** (surface form + char offsets) so every extraction can be audited.
- **Entity resolution support**: canonical `entities` + `entity_aliases` with a
  `gin_trgm_ops` index for fuzzy candidate lookup; soft-merge via `merged_into`.
- **Graph views** (`v_graph_nodes`, `v_graph_edges`) hide merged entities and are
  the only thing NetworkX reads. Community precedent: keep tables authoritative and
  expose derived graph structures ([evokoa/pggraph](https://github.com/evokoa/pggraph) docs make the same split).
- Real-world reference design (news → entity links → NetworkX) validated the
  "edge table + association table for idempotency + NetworkX loader" approach:
  [martinsnygg/stratintel PR #352](https://codeberg.org/martinsnygg/stratintel/pulls/352).

---

## 3. Entity & relation extraction from unstructured documents

### Landscape (2026)
| Approach | Strengths | Weaknesses | Role here |
|---|---|---|---|
| **GLiNER2** (fastino) | Zero-shot: NER + text classification + **structured extraction + relation extraction** in one forward pass; small encoder, runs locally; EMNLP 2025 demo paper | Needs torch; labels must be designed | **Primary extractor** |
| spaCy | Fast, mature, great tokenizer/parser; easy custom rules | Closed label sets; weaker zero-shot | Sentence splitting, fallback NER |
| Stanza / Flair | Solid classical NER, multilingual | Heavier, no relation extraction | Not used |
| Relik / REBEL | Entity linking + relation extraction, knowledge-graph focused | Extra models to host | Candidate later for linking |
| LLM (GPT/Claude/etc.) | Best zero-shot flexibility, explanations, schema-following | Cost/latency at scale; hallucination risk | **Escalation only**, and for rare relation types |
| Heuristic baseline | No deps, deterministic, testable | Low precision/recall | Smoke tests + pipeline bring-up |

Sources: [GLiNER2 GitHub](https://github.com/fastino-ai/GLiNER2),
[GLiNER2: Efficient Multi-Task IE (EMNLP 2025)](https://aclanthology.org/2025.emnlp-demos.10/),
[GLiNER2 overview — Towards Data Science](https://towardsdatascience.com/gliner2-extracting-structured-information-from-text/),
[origin story (NER → structured extraction)](https://fastino.ai/blog/gliner2-5-origin-story),
[Relik with LlamaIndex — Neo4j](https://medium.com/neo4j/entity-linking-and-relationship-extraction-with-relik-in-llamaindex-ca18892c169f),
[Neo4j agent-memory extraction pipeline notes](https://neo4j.com/labs/agent-memory/explanation/extraction-pipeline/) ("extraction is an inference step — a well-formed result can still omit or invent information").

### The effective pipeline (what we built)
The research consistently shows **extraction alone is not the hard part** —
*filtering and resolving* are. Recommended (and implemented) shape:

1. **Ingest** documents (txt/md now, PDF via pypdf; docling/unstructured are the
   heavier alternatives if layout-heavy PDFs arrive).
2. **Candidate generation** (recall-oriented): GLiNER2 (or spaCy/baseline) proposes
   entity mentions with types + candidate relations with evidence sentences.
3. **Typed decisions** (precision-oriented, cheap): see §4 — Jev filters/arbitrates.
4. **Entity resolution**: alias/trigram blocking in SQL → Jev alignment verdicts →
   merge/curate. **Merging wrongly is more expensive than missing a match**, so the
   decision must have a middle outcome (see §4 cookbook).
5. **Store** reified facts with provenance (schema §2).
6. **Project** into NetworkX for analytics/visualization.

### Escalation policy (implemented 2026-10-11, P4.3/P4.5, D11)
- **spaCy** is the lightweight fallback: `en_core_web_sm` NER + dependency-path
  relations (lowest-common-ancestor verb, subject → object). Fast and local, but
  the small model's NER is noisy and relation recall is low — a fallback, not a
  primary.
- **LLM escalation is opt-in** (`EXTRACTION_ESCALATE` / `--escalate`), matching
  D2's "escalate only when uncertain" cost model. The cheap pass triggers
  escalation when it finds no mentions, no relations, mostly low-confidence
  mentions, or sparse relations (< 0.25 relations/mention). The LLM (OpenRouter
  chat completions, `openai/gpt-4o-mini` by default) returns strict JSON that is
  anchored back to char offsets.
- **Jev still guards the LLM.** Live run: the LLM proposed
  `GreenGrid Energy FOUNDED Dr. Maria Chen` (reversed); Jev rejected it
  (noul 0.05). Precision filtering stays cheap and calibrated regardless of which
  extractor produced the candidate.

---

## 4. Jev (SystemOne) — what it is and how it facilitates extraction

### What it is
**Jev** is TypeSafe's flagship **"System One" model**: like an LLM it understands
natural language, but it **returns typed decisions and calibrated probabilities
instead of generated text**. You define the answer space; it answers:

| Primitive | Question it answers | Example (ours) |
|---|---|---|
| `Choice` | Which of these options? | "Which entity type: PERSON / ORG / PLACE?" |
| `Score` | Ordinal level on a labeled rubric | "How likely are these two mentions the same entity? 0=different, 1=uncertain, 2=same" |
| `Noul` | Probability of true (0–1) | "Does this sentence assert X works for Y?" |

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`, model `jev-latest`; accepts
  strings, JSON objects, and arrays of text (text-only input).
- Probabilities are **calibrated against outcomes** (calibration is statistical, not
  a per-answer guarantee).
- Sources: [docs.typesafe.ai — System One](https://docs.typesafe.ai/concepts/system-one),
  [API guide](https://typesafe-jev.com/en/guides/api/),
  [innfactory model review](https://innfactory.ai/en/ai-models/typesafe-jev/),
  [awesome-typesafe-jev field guide](https://github.com/AbdelStark/awesome-typesafe-jev).

### How it facilitates *entity extraction* specifically
Jev does **not** find spans (that's GLiNER2/spaCy's job). It sits *after* candidate
generation and makes the thousands of small judgments a knowledge-graph build
needs — fast, cheap, calibrated, with **no prose to parse**:

1. **Mention filtering** — `Noul`: "Is '<span>' a real-world entity reference
   rather than a fragment, title, or artifact?"
2. **Type arbitration** — `Choice`: when two extractors disagree on the type of a
   mention, ask once with the union of proposed types as options.
3. **Entity alignment / resolution** — this is the killer feature, and TypeSafe
   publishes an official cookbook for it:
   [Cookbook: Knowledge graph entity alignment](https://jevwiki.ai/wiki/cookbooks/entity-alignment.md).
   Pattern (we implement it in `entity_graph/extract/jev.py::align_pair`):
   - Put both entities in **one state** as `entity_a` / `entity_b`.
   - Ask **one `Score` question whose three labeled criteria levels are the three
     outcomes**: `0` = different entity → leave unlinked; `1` = related-but-uncertain
     → curator queue; `2` = same entity → assert merge.
   - **Route by rounding the score to the nearest level — no threshold constants.**
   - Add **`Noul` sidecar questions per field** (e.g. same name? same kind?) in the
     same request so a curator sees *which* fields disagree.
   - Rationale from the cookbook: merging inappropriately is the more expensive
     mistake, so a middle outcome must exist (a `Choice` would lose the ordering; a
     bare `Noul` + threshold loses the labeled middle).
4. **Relation validation** — `Noul`: "Does this evidence sentence assert the
   relation (X, WORKS_FOR, Y)?" Drop or down-weight relations that fail.
5. **Fact confidence calibration** — store the Jev probability as the fact's
   `confidence` column, giving comparable confidences across extractors.

### Local/no-cost development path
[codesoda/systemone](https://github.com/codesoda/systemone) (MIT) is a CLI +
**Jev-compatible HTTP API** that can serve multiple backends — **OpenJev, Laya,
Kev, GLiNER2, or hosted TypeSafe Jev** — so the same client code can run against
GLiNER2 locally during development and against hosted Jev in production by
changing config. (Young project — 2026-09; treat as dev convenience, not a
dependency.) Community catalog of tools/demos: [awesome-typesafe-jev](https://github.com/AbdelStark/awesome-typesafe-jev).

### Maturity caveats
- Young ecosystem (TypeSafe Jev documented 2026; wikis/community guides active).
- Text-only input; no long-document reasoning (keep states small: two entity
  profiles, one evidence sentence).
- **Action item (P5.6)**: verify exact request/response field names against the
  live API once a key exists; our client is written from the documented contract
  and offline examples.

### Verdict
**Yes — adopt Jev, scoped to decisions.** The most effective overall method is the
hybrid: GLiNER2 for recall (spans + candidate relations), Jev for calibrated
judgment (filter/align/validate), LLM only for the residual hard cases. This is
also the cheapest architecture at scale: Jev-style decision calls are far cheaper
than LLM calls per judgment.

---

## 4b. Assessment (2026-10-10): can Jev replace GLiNER2 as the extractor?

Trigger: `typesafe/jev-1.13` on the OpenRouter decisions API, and the question
*"is Jev a universal classifier we can use instead of GLiNER2?"*.

**Short answer: no — not as the extractor.** Jev is a typed **classifier/decision**
model: it *selects* among options the code provides and cannot discover spans.
The only Jev-only route to NER is to generate candidates in code (regex or a
segmenter) and let a `Choice` pick/verify them.

Evidence:
- TypeSafe's own guidance: *"Jev is not trained to generate text… For data
  extraction, it is better to extract possible options using regex or a
  generative model and let `jev-1.13` pick the correct extraction."* There is no
  field for a value it discovered.
  ([Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13))
- Measured KG-extraction study (William Lyon, 2026-09-17): GLiNER2.5 produced 170
  candidate edges in ~14 s on a CPU with **no key**; *"a System One model can't
  find an edge; it has no spans and nothing to anchor."* Jev's measured value was
  as a **judgment layer**: an assertion gate moved precision 0.279 → 0.404 at
  *identical* recall, and unioning relation *selection* moved F1 0.330 → 0.442.
  Conclusion: *"Jev is not a replacement for either of the other extractors."*
  ([blog](https://lyonwj.com/blog/typesafe-jev-knowledge-graph-extraction))
- **JevSpan** (PyPI `jevspan`) is the exception that proves the rule: it turns
  Jev into zero-shot NER by segmentation → candidate windows as `Choice` options →
  verification → boundary polishing, reporting **73.7 strict F1** across 12
  benchmarks with no training. But *"spans come from the segmenter. Jev only
  scores candidates, so an entity that no window covers cannot be found,"* and
  *"every document goes through the Jev API."*
  ([PyPI](https://pypi.org/project/jevspan/))

**What Jev is for here (unchanged from D3):** mention filtering (`Noul`), type
arbitration (`Choice`), entity alignment (3-level `Score` + `Noul` sidecars),
relation validation (`Noul`), and fact-confidence calibration.

**Cost/latency vs GLiNER2.** Jev is hosted (API key + network): ~\$0.42/M input
tokens (~\$0.001 per decision) at TypeSafe, ~70–500 ms per request, many questions
evaluated in parallel against one state. GLiNER2 runs locally, free, no key. A
Jev-only NER pipeline also pays several calls per document (nominate / verify /
polish / propagate).

**Verdict.** Keep a local span model (GLiNER2) as the candidate generator and Jev
as the calibrated judgment layer ([D3](DECISIONS.md#d3)). Revisit a JevSpan-style
candidate+`Choice` NER backend only if dropping the torch install becomes a hard
requirement *and* an API key, network egress, and per-document cost are
acceptable ([D7](DECISIONS.md#d7)).

### Verified Jev API contract (2026-10-10)
- Endpoint `POST https://api.typesafe.ai/v1/systemone`, body `{model, state,
  questions}`. OpenRouter mirrors it at
  `POST https://openrouter.ai/api/alpha/decisions` with `model:
  typesafe/jev-1.13`.
- Every question uses **`instructions`** (string | object | array).
  `choice.criteria` is a **map** option → description (2–255, no `options`
  array); `score.criteria` is an **ordered array** of 2–10 levels (no `levels`
  field); `noul.criteria` is optional `{true, false}`.
- Response `{model, answers: {id: {...}}, usage}`: `noul` → `{type, noul}`
  (no confidence); `choice` → `{type, choice, probabilities, confidence}`;
  `score` → `{type, score, legend, probabilities, confidence}`.
- Our client had used `question`/`options` and parsed answers as scalars; fixed
  to `instructions`/`criteria` and the typed answer fields.

### Live verification (2026-10-11, P5.6) — confirmed
A live smoke test through the OpenRouter mirror (`model` resolved to
`typesafe/jev-1.13-20260917`, provider TypeSafe) returned exactly the documented
shape for all three question kinds:

```
noul   -> {"type":"noul","noul":0.9}
choice -> {"type":"choice","choice":"billing",
           "probabilities":{"sales":0,"support":0.11,"billing":0.89},
           "confidence":0.83}
score  -> {"type":"score","score":3,
           "legend":{"0":"...","3":"angry: strong dissatisfaction..."},
           "probabilities":{...},"confidence":1}
```

Notes: (a) `score.legend` is an object keyed by level *index* (strings), not an
array; (b) cost was ~$0.00002 per small call (`usage.cost`), so batched
validation over a document is negligible; (c) the response also carries `id`
and `provider`. This confirms the client contract end to end — no field
mismatches. The entity-alignment Score routes cleanly by `round(score)` (e.g.
`Maria Chen` vs `Dr. Maria Chen` → 1.84 → merge), and batched Noul relation
validation agrees with single-call validation on known-true/false cases.

### Pratt router mirror (2026-10-11, D12)
The Pratt model router at `https://llm.pratt.edu/v1` is OpenAI-compatible
(`/v1/chat/completions`, `/v1/responses`, `/v1/models`) and also fronts the Jev
decision model. Its decisions surface is **`POST /v1/systemone`** with
`model: pratt-jev` (the same `{state, questions}` body), returning
`{model, answers, usage, id, provider}` with `usage.cost`. Verified live:

```
align  pratt-jev -> score 1.84 (merge) for a near-duplicate ORG pair
noul   pratt-jev -> 0.99 for "Maria Chen FOUNDED GreenGrid Energy",
                    0.02 for its reversed form
```

`pratt-jev` is a *decisions* model: `/v1/chat/completions` and `/v1/responses`
reject it ("…decisions model … use the /api/alpha/decisions endpoint instead"),
so it is wired only to `/v1/systemone`. The general-purpose model
`pratt-medium` is used for LLM escalation (`/v1/chat/completions`).

Sources: [docs.typesafe.ai/models](https://docs.typesafe.ai/models),
[docs.typesafe.ai/api](https://docs.typesafe.ai/api),
[jev-1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13),
[jev-api.org/docs](https://jev-api.org/docs),
[lyonwj.com KG extraction study](https://lyonwj.com/blog/typesafe-jev-knowledge-graph-extraction),
[jevspan](https://pypi.org/project/jevspan/),
[OpenRouter jev-1.13](https://openrouter.ai/typesafe/jev-1.13).

---

## 4c. Addendum (2026-10-11): JointIE proposes, a modality gate disposes

Independent `.relations()` cannot enforce typed endpoints. GLiNER2.5 `JointIE`
([tutorial](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/15-joint_ie.md))
decodes entities and relations together under head/tail types, uniqueness, and
`no_self_loops()`. That is now the candidate generator ([D13](DECISIONS.md#d13)).

The assertion Noul does not distinguish a completed fact from a forecast or a
denial. William Lyon's
[Decide study](https://lyonwj.com/blog/gliner-decide-knowledge-graph-extraction)
(2026-09-24), on the same propose/dispose split as the Jev post, measured:

| System | precision | recall | F1 |
|---|---|---|---|
| GLiNER2.5, ungated | 0.279 | 0.404 | 0.330 |
| Jev assertion gate | 0.404 | 0.404 | 0.404 |
| Decide modality + relation agreement | 0.388 | 0.404 | 0.396 |

Decide is a verifier, not an extractor and not a pair comparator. Short labels
and an argmax beat a threshold on P(fact); a shared prompt shifts the two
heads, so they are separate passes. Jev still wins on instruction-following
pair comparison. This repo therefore asks Jev for modality *beside* the Noul
when a key exists, and uses Decide only as the offline gate
([D14](DECISIONS.md#d14)). The 0.5 Noul cutoff is not retuned until
`data/gold/` is scored ([D15](DECISIONS.md#d15)).

## 4d. Local Clef-Flash vs Jev (2026-10-11)

A local **SystemOne-compatible** server is reachable on this network:
"Clef-Flash (hardened)" v0.2.0 (`Cloudflare/clef-flash`, CUDA) at
`http://192.168.1.242:8001`, exposing `/v1/systemone` and `/predict`
(OpenAPI at `/openapi.json`). Its response body matches Jev's exactly —
`answers[id]` with `type` / `score` / `noul` / `choice` / `probabilities` /
`confidence` / `legend` — so `SystemOneClient` drives it unchanged; only
`SYSTEMONE_URL`/`SYSTEMONE_MODEL` (and any non-empty key) change.

`scripts/compare_providers.py` runs both providers over the gold fact set
(17 items: 9 facts, 2 nonfacts, 6 traps) and 7 alignment pairs:

| Metric | Jev (`pratt-jev`) | Clef-Flash (local) |
|---|---|---|
| gate accuracy (promote == role) | 0.941 | 0.706 |
| leaks (nonfact/trap promoted) | 0 | 5 |
| missed facts | 1 | 0 |
| alignment intended-match | 3/7 | 2/7 |
| relation batch wall time | 1.04 s | 2.12 s |

Clef-Flash assigns a high assertion Noul to falsehoods — reversed `FOUNDED`
0.925, fabricated `SUBSIDIARY_OF` 0.543, "invest" evidence for `INVESTED_IN`
0.711 labelled `fact` — so it promotes 5 of the 8 nonfact/trap items. Jev is
far more precise on negation and attribution, but rejected one true fact (a
past-employment `WORKS_FOR`, Noul 0.24). Both route canonical+alias pairs to
`curator` rather than `merge` (the conservative middle level).

### 4d-i. Calibrating the prompt (2026-10-11)

The model is fixed; only the *wording* and *request shape* can change. A
calibration harness (`scripts/calibrate_clef.py`) sweeps prompt variants against
the same gold set, and the winning variant is captured as an opt-in
`RelationPromptProfile` (`extract/profiles.py`, D17). Three levers mattered, in
order of effect:

1. **Request batch size is dominant.** A small model's answers degrade as more
   claims share one context. The same `tense` wording scored 0.824 at 12 claims
   per request, 0.882 at 4, and **0.941 at 1–2**. Queried alone, the two worst
   items flip correctly (forecast → `forecast` 0.79; reversed `FOUNDED` → Noul
   **0.07**). The calibrated profile therefore sends one claim per request.
2. **Modality criteria must name the tense cues.** Spelling out
   `will / plans to / is to / expansion into` as `forecast` stops the model
   reading the main verb ("cooperate") as present and missing the modal ("will").
   This closed the two forecast-as-fact leaks (`LOCATED_IN Poland`,
   `INVESTED_IN`).
3. **The assertion must name the doer/receiver.** Asking "is `subject` the doer
   and `object` the receiver …" (with an explicit passive-voice warning) drops
   the reversed `FOUNDED` trap from Noul 0.92 to 0.35, at the cost of one
   fragment-evidence fact. A separate `direction` Choice question was tried and
   **rejected**: the model misreads locative relations (`LOCATED_IN`) as
   "object_to_subject".

| Metric | Jev (`pratt-jev`) | Clef-Flash (default prompts) | Clef-Flash (calibrated `clef` profile) |
|---|---|---|---|
| gate accuracy (promote == role) | 0.941 | 0.706 | **0.941** |
| leaks (nonfact/trap promoted) | 0 | 5 | **0** |
| missed facts | 1 | 0 | 1 |
| promote-decision agreement w/ Jev | — | 11/17 | **15/17** |
| relation batch wall time | 1.04 s | 2.12 s | 4.77 s (17 sequential) |

Calibrated Clef-Flash **matches Jev on the gate** (0.941, 0 leaks) — but it
misses a *different* fact (`PARTNERED_WITH` with fragment evidence, Noul 0.42),
so it is a peer, not a superset. Alignment is untouched by the relation profile
and stays 2/7. The cost is throughput: one claim per request means ~2.5× the
wall time for the same items.

**Verdict (revised).** Clef-Flash is a drop-in transport, and — with the `clef`
profile and single-claim requests — a drop-in *gate* on this gold set, at the
price of sequential calls. Keep Jev as the default; select the calibrated
profile with `SYSTEMONE_PROFILE=clef` when running against the local server.

## 5. Document ingestion
- **txt/md**: direct. **PDF**: `pypdf` for text-layer PDFs (implemented, optional).
- For scanned/layout-heavy PDFs, the current best tools are IBM **docling** and
  **unstructured.io** (both heavier installs) — deferred until such documents
  actually appear. ([docling](https://github.com/DS4SD/docling),
  [unstructured](https://github.com/Unstructured-IO/unstructured))

### OCR preprocessing before extraction (2026-10-11, P3.4, D16)
Published practice for feeding OCR output to NER agrees on a small, mostly
rule-based cleanup pass rather than model-specific tricks: strip page/layout
artifacts, rejoin lines broken mid-sentence, remove footnote/reference markers,
de-hyphenate soft line breaks, and normalise Unicode punctuation. The HAL
study *When to Use OCR Post-correction for Named Entity Recognition?* finds the
useful gains come from exactly these structural fixes, and that aggressive
character-level correction can hurt (it changes names). Our Shanghai book run
confirmed the shape of the problem: 262/334 pages end mid-sentence, 220
`\textsuperscript{N}` markers, and ~14 hyphen breaks — but 3 of those are real
hyphenated romanisations, so blanket de-hyphenation is wrong.

Implementation (`preprocess.py`, D16) uses the **document itself as the
de-hyphenation dictionary**: `Shang-\nhai` → `Shanghai` because "Shanghai"
occurs elsewhere, while `Yueh-sheng` keeps its hyphen because the joined form
does not. No bundled word list is required, and 13/14 real cases resolved.
Sources: [Labelyourdata — OCR preprocessing for NLP](https://labelyourdata.com/articles/ocr-data-entry),
[When to Use OCR Post-correction for NER? (HAL)](https://hal.science/),
[Mastering Document Preprocessing for NER with BERT](https://medium.com/@mleinbayarea/mastering-document-preprocessing-for-ner-with-bert-a9f54d9a2c72).

## 6. Postgres ↔ NetworkX integration
- Load with pandas: `pd.read_sql("SELECT * FROM v_graph_edges", conn)` →
  [`nx.from_pandas_edgelist`](https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html)
  into a `nx.MultiDiGraph` (multiple relation types per pair are expected here);
  node attributes from `v_graph_nodes`.
- Analytics available natively in NetworkX ≥3.x: `degree_centrality`,
  `betweenness_centrality`, `eigenvector_centrality`,
  `nx.community.louvain_communities`, `nx.k_core`, shortest paths.
- Scale note: NVIDIA's accelerated NetworkX backend exists if graphs grow into
  millions of edges ([NVIDIA dev blog](https://developer.nvidia.com/blog/accelerated-production-ready-graph-analytics-for-networkx-users/)) — not needed now.

## 7. Visualization
- **pyvis** — interactive HTML with type-based coloring (chosen; optional extra).
- matplotlib + `nx.draw_spring` for static figures in reports.
- Deferred: Graphistry/Cytoscape/Memgraph consoles — unnecessary for v1.
