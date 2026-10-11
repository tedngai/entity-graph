# DECISIONS.md — Decision Records

Format: lightweight ADRs. Status: proposed | accepted | superseded.

---

## D1 — Graph storage: relational edge tables + recursive CTEs (v1); Apache AGE deferred
**Status:** accepted · **Date:** 2026-10-04

**Context.** Entities/relationships must live in PostgreSQL (project constraint).
Options: plain edge tables, Apache AGE (openCypher extension), Postgres-native
graph layers (`pg_ripple`, `post-graph`, `pggraph`), or a separate graph DB.

**Decision.** Plain relational tables (`entities`, `relationships`, `facts`,
`mentions`) with `pg_trgm` fuzzy indexes and graph *views* consumed by NetworkX.
Add Apache AGE only if Cypher-style traversal becomes a real, recurring need.

**Why.** Zero operational overhead, full SQL/JSONB tooling, and NetworkX performs
the analytics anyway — a graph query language in the DB duplicates that.
Recursive CTEs cover bounded traversals. Sources: `docs/RESEARCH.md` §2.

---

## D2 — Extraction: GLiNER2 primary; spaCy fallback; LLM escalation only
**Status:** accepted · **Date:** 2026-10-04

**Decision.** Candidate generation uses **GLiNER2** (zero-shot schema-conditioned
NER + classification + relation extraction, one forward pass, runs locally).
**spaCy** provides sentence splitting and a fallback NER path. An LLM is used
only to escalate documents/spans the cheap pass flags as uncertain.

**Why.** Research (EMNLP 2025 GLiNER2 paper; practitioner write-ups; Neo4j
extraction-pipeline notes) shows a recall-first extractor + cheap precision
filtering beats one expensive monolithic extractor on cost and auditability.
A dependency-free baseline extractor exists for tests and bring-up.

---

## D3 — Use Jev (SystemOne) for calibrated typed decisions in the pipeline
**Status:** accepted · **Date:** 2026-10-04

**Context.** User asked to evaluate "jev (systemone)" methods for entity
extraction. Research confirmed: Jev (TypeSafe) is a *System One* decision model —
returns typed `Choice`/`Score`/`Noul` answers with calibrated probabilities, no
generated text. It does **not** do span detection.

**Decision.** Adopt Jev for the judgment layer of the pipeline:
1. entity alignment / dedup via the official 3-outcome `Score` cookbook
   (separate → curator → merge; route by rounding; **no thresholds**; `Noul`
   sidecars expose which fields disagree),
2. mention filtering (`Noul`),
3. type arbitration between disagreeing extractors (`Choice`),
4. relation validation against evidence sentences (`Noul`),
5. fact confidence calibration (Jev probabilities → `facts.confidence`).

**Why.** The vendor cookbook addresses exactly this use case (knowledge-graph
entity alignment); decision calls are cheap and fast relative to LLM calls;
typed outputs require no output parsing. `codesoda/systemone` provides a
Jev-compatible local adapter (can serve GLiNER2) for development.

**Caveats.** Young ecosystem; text-only states (keep them small); verify live
API field names when a key is available (PROGRESS P5.6).

---

## D4 — Reified facts with provenance; relationships are an aggregate
**Status:** accepted · **Date:** 2026-10-04

**Decision.** Store claims in `facts` (statement, evidence quote, document,
extractor, confidence, `valid_from`/`valid_to`, `is_current`). `relationships`
is the unique aggregated edge table NetworkX consumes. Never hard-delete:
superseded facts flip `is_current`; duplicate entities get `merged_into`.

**Why.** "Key facts" need evidence and validity to be trustworthy; aggregates
alone lose provenance, and per-claim storage enables re-derivation, temporal
analysis, and per-extractor quality comparison.

---

## D5 — NetworkX reads only the graph views; pyvis for interactive output
**Status:** accepted · **Date:** 2026-10-04

**Decision.** `graph_store.load_graph()` is the only code path from Postgres to
NetworkX (`v_graph_nodes`/`v_graph_edges` → pandas → `nx.MultiDiGraph`).
Visualization via pyvis HTML (optional `[viz]` extra), matplotlib for statics.

**Why.** One load path keeps merged-entity handling in SQL; MultiDiGraph fits
multiple relation types per pair; pyvis gives zero-server interactive graphs
that are easy to share.

---

## D6 — Long documents: overlapping sentence windows with rebased offsets
**Status:** accepted · **Date:** 2026-10-10

**Context.** Extraction backends (GLiNER2 in particular) degrade or blow up on
very long inputs, and one window over a whole document gives the model no
locality. Splitting naively, though, breaks the provenance invariant: mention
char offsets must index the *full* document text.

**Decision.** `chunking.py` splits documents into windows of up to
`CHUNK_MAX_SENTENCES` / `CHUNK_MAX_CHARS` with `CHUNK_OVERLAP_SENTENCES` shared
sentences between neighbours. Each `Chunk` records absolute `(start, end)`
offsets (`chunk.text == source[start:end]`). `pipeline.extract_document` runs the
backend per window, rebases chunk-relative mention offsets to document
coordinates, and dedupes overlap (mentions by span, relations by
`(source, target, relation)`, keeping the highest confidence). Single-window
documents skip chunking. Chunking is pure stdlib and backend-agnostic.

**Why.** Windows bound input size and give locality; overlap prevents losing
mentions/evidence that straddle a boundary; rebasing keeps offsets auditable
against `documents.raw_text`; dedupe keeps the overlap from inflating the graph.

**Caveats.** Sentence splitting is heuristic (handles common abbreviations and
initials, not all cases). Overlap dedupe collapses multiple same-edge evidences
within a document to one proposal, matching the baseline extractor's behaviour;
if per-sentence evidence multiplicity is wanted later, key relations by evidence
instead.

---

## D7 — Jev is not the extractor; keep a local span model for candidates
**Status:** accepted · **Date:** 2026-10-10

**Context.** Jev 1.13 (`typesafe/jev-1.13`) shipped on the OpenRouter decisions
API, and we were asked whether Jev — "a universal classifier" — could replace
GLiNER2 as the entity/relation extractor, before investing in the GLiNER2 backend.

**Decision.** Do **not** use Jev as the primary extractor. Keep a local span model
(GLiNER2) for candidate generation (mentions + relation proposals) and use Jev
strictly as the calibrated judgment layer (D3). A Jev-only NER path is possible
only via candidate generation in code (regex/segmenter) + `Choice` selection and
verification, as JevSpan does; it is deferred unless dropping the torch install
becomes a hard requirement *and* an API key, network egress, and per-document cost
are acceptable.

**Why.** Jev returns typed decisions, not spans — it "has no field for a value it
discovered," so it can select among candidates but cannot find them. TypeSafe's
own guidance says to over-find candidates with regex/a generative model and let
Jev pick. An independent measured study found GLiNER2.5 generated 170 candidate
edges in ~14 s locally with no key while Jev added only judgment value (assertion
gate: precision 0.279 → 0.404 at unchanged recall), concluding "Jev is not a
replacement for either of the other extractors." JevSpan reaches 73.7 strict F1
zero-shot but its spans come from a punctuation segmenter and every document hits
the API. See [RESEARCH.md §4b](RESEARCH.md#4b-assessment-2026-10-10-can-jev-replace-gliner2-as-the-extractor).

**Consequences.** The GLiNER2 backend (P4.4) remains the primary extraction path.
While assessing this we verified the Jev HTTP contract against the published API
reference and corrected `extract/jev.py` (`instructions`/`criteria`, typed answer
fields), partially advancing P5.6; a live smoke test still needs a key.

---

## D8 — GLiNER2 backend: `AutoExtractor` + `gliner2.5-base-v1`; CUDA torch from the PyTorch index
**Status:** accepted · **Date:** 2026-10-10

**Context.** Implementing P4.4 (primary extractor) on this Windows machine. The
existing skeleton targeted the legacy `GLiNER2.from_pretrained` + `add_entities`/
`add_structured` API, which no longer matches the installed package.

**Decision.**
1. Load with `AutoExtractor.from_pretrained(...)` and default model
   `fastino/gliner2.5-base-v1` (194M, DeBERTa-v3-base). `AutoExtractor` dispatches
   both the span and boundary architectures; `GLiNER2.from_pretrained` is the
   legacy span loader and will not dispatch GLiNER2.5 checkpoints.
2. Build the schema with `create_schema().entities([...]).relations([...])` and
   call `extract(text, schema, include_confidence=True, include_spans=True)`.
   Result shape: `{"entities": {label: [{text, confidence, start, end}]},
   "relation_extraction": {rel: [{head, tail, confidence}]}}`.
3. Device from `GLINER_DEVICE` (`auto` → CUDA if available, else CPU). Model,
   device and threshold come from `GLINER_MODEL` / `GLINER_DEVICE` /
   `GLINER_THRESHOLD`.
4. Install with `gliner2[local]`. On Windows, PyPI resolves a **CPU-only** torch,
   so CUDA requires `pip install torch --index-url
   https://download.pytorch.org/whl/cu130` afterwards.

**Why.** Verified on the RTX 2000 Ada (8.6 GB, sm_89, torch 2.14.1+cu130): model
loads in ~5.5 s and extracts the bundled sample in ~0.69 s, returning 41 mentions
and 27 relations with exact char offsets (baseline gave far fewer, noisier
mentions). No extra dependency beyond the torch stack.

**Consequences.** `pyproject.toml` `[extraction]` now lists `gliner2[local]` (it
previously listed `gliner2-gui`, excluded on Windows, and did not actually install
the extractor). The GLiNER2 backend remains the primary path per D2/D7; low-
confidence relations it emits (e.g. `WORKS_FOR` at 0.44) are exactly the
candidates Jev validation (D3) is meant to filter.

---

## D9 — Jev integration: alignment routing, batched Noul validation, OpenRouter inference
**Status:** accepted · **Date:** 2026-10-11

**Context.** Implementing P5.3–P5.6 with a live key (OpenRouter mirror). Two
questions had to be settled: how to select the endpoint/credentials, and how to
use Jev without making a call per candidate.

**Decision.**
1. **Credentials/endpoint inference.** `SYSTEMONE_API_KEY` wins; a bare
   `OPENROUTER_API_KEY` selects the OpenRouter mirror
   (`https://openrouter.ai/api/alpha/decisions`, model `typesafe/jev-1.13`) with
   optional `HTTP-Referer` / `X-OpenRouter-Title` headers. `SYSTEMONE_URL` /
   `SYSTEMONE_MODEL` override either default.
2. **Entity alignment (P5.3).** Exact and alias hits short-circuit. Only a
   *fuzzy* (pg_trgm) candidate is sent to Jev's Score question and routed by
   `round(score)`: **merge** links + records a `source='jev'` alias;
   **curator** creates a new entity and writes the pair to `curator_queue`;
   **separate** creates a new entity. A failed call falls back to the previous
   deterministic trigram behaviour.
3. **Relation validation (P5.4).** All pending proposals are Noul-scored in
   **batched** calls (≤25 claims per request, state = list of claims, one
   question per claim id). Proposals with `noul < 0.5` are marked `rejected`
   (soft, never deleted) instead of promoted; the score is stored in
   `relation_proposals.validation`. `--no-validate` skips this.
4. **Curator queue (P5.5).** New `curator_queue` table; `entity-graph curator`
   lists pending pairs. `relation_proposals.is_promoted` became
   `status` (pending | promoted | rejected) plus `validation`.

**Why.** A live test confirmed the contract (RESEARCH §4b) and showed batching
does not degrade Noul accuracy (single vs batched agreed on known cases), at
~$0.00002/call. Routing fuzzy matches through Jev rather than a bare similarity
threshold avoids over-merging (risk R3). On the sample: GLiNER2 produced 14
relation candidates; Jev promoted 9 (all correct) and rejected 5 genuine
falsehoods (`ACQUIRED` vs "invest", two fabricated `SUBSIDIARY_OF`).

**Consequences.** `db/schema.sql` gains `curator_queue` and the proposal
`status`/`validation` columns (re-run `init-db`). Jev remains optional: with no
key the pipeline degrades to exact/alias/trigram and promotes on extractor
confidence.

---

## D10 — Sentence boundaries at paragraph breaks, not single newlines
**Status:** accepted · **Date:** 2026-10-11

**Context.** With GLiNER2 + Jev wired end to end, Jev correctly *rejected* a
true relation (`GreenGrid Energy PARTNERED_WITH Nordwind Capital`). The cause was
ours, not Jev's: `chunking.split_sentences` treated **every newline** as a
sentence boundary, so the hard-wrapped sample (wrapped at ~78 cols) became one
"sentence" per line. Chunks split mid-sentence and relation evidence was quoted
from only the first wrapped line — which did not contain the object.

**Decision.** `split_sentences` now treats a newline as a boundary only at a
**paragraph break** (a whitespace run with ≥2 newlines); a lone newline is
intra-sentence whitespace. Spans still tile the text exactly, so all char
offsets (invariant #1) are unchanged. Additionally, `gliner_extractor` binds
relation evidence across the head→tail sentence range rather than the head's
sentence alone.

**Why.** This is the common case for ingested prose/PDFs (hard-wrapped lines).
After the fix the sample chunked into 2 windows instead of 3, the true
`PARTNERED_WITH` relation passed Jev (noul 0.98), and total Jev rejections
dropped while precision rose.

**Consequences.** Titles and single-newline line lists are treated as one
sentence, which is acceptable for prose and bounded by `max_chars`. New tests
cover hard-wrapped prose, paragraph breaks, and head→tail evidence.

---

## D11 — spaCy is the dependency-based fallback; LLM escalation is an opt-in wrapper
**Status:** accepted · **Date:** 2026-10-11

**Context.** Implementing P4.3 (spaCy) and P4.5 (LLM escalation). D2 already
fixes the roles: spaCy is a lightweight fallback, an LLM is used only to escalate
uncertain documents. The open questions were *how* to extract relations with
spaCy and *when* to spend on an LLM.

**Decision.**
1. **spaCy backend.** `en_core_web_sm` NER mapped to our vocabulary, plus
   dependency-based relations: for each entity pair in a sentence, take the
   lowest common ancestor; if it is a verb and both entities are its arguments
   (direct or one preposition hop), map the lemma to a relation (`found` →
   `FOUNDED`, `invest` → `INVESTED_IN`, …), with the subject as source and the
   object as target. Pairs without exactly one subject + one object are dropped;
   results are deduplicated. Missing model → actionable install hint.
2. **LLM backend (`llm`).** OpenAI-compatible chat completions (OpenRouter by
   default, `openai/gpt-4o-mini`). The model is asked for a strict JSON document;
   the reply is parsed defensively (code fences tolerated) and every entity
   surface is anchored back to char offsets — unanchorable items are dropped so
   invariant #1 always holds. Empty relation evidence falls back to the source's
   sentence.
3. **Escalation wrapper.** `EscalatingExtractor` wraps any non-`llm` backend and
   is **opt-in** (`EXTRACTION_ESCALATE=true` or `--escalate`). It escalates when
   the cheap pass found no mentions, found no relations, returned mostly
   low-confidence mentions, or produced sparse relational structure
   (< 0.25 relations per mention). Results are merged: every cheap-pass item is
   kept and LLM items not already present (mentions by span, relations by
   `(source, target, relation)`) are added.
4. `get_extractor(backend, config)` applies the wrapper.

**Why.** D2's cost model only holds if escalation is rare and explicit; making it
opt-in guarantees the default pipeline never spends on LLM calls. The triggers
are observable properties of the cheap result, not guesses about the text.

**Consequences.** New config knobs `LLM_MODEL`, `LLM_URL`, `LLM_API_KEY` (falls
back to `OPENROUTER_API_KEY`), `EXTRACTION_ESCALATE`, `ESCALATION_CONFIDENCE`.
Verified live: the `llm` backend runs the full pipeline (17 mentions → Jev
promoted 10 edges, and Jev *rejected* the LLM's reversed `FOUNDED` at noul 0.05 —
the judgment layer still guards the expensive path); `spacy --escalate` triggered
escalation on the sample and merged LLM relations. `en_core_web_sm` NER is
noisier than GLiNER2 (types like `Nordwind Capital` → PERSON), which is expected
for the fallback and why GLiNER2 remains primary.

---

## D12 — Pratt router (`llm.pratt.edu`) is the default LLM + Jev provider
**Status:** accepted · **Date:** 2026-10-11

**Context.** The machine has access to the Pratt model router at
`https://llm.pratt.edu/v1` (OpenAI-compatible), keyed by `PRATT_LLM_API_KEY` in
`.env`. It fronts both a local general-purpose model (`pratt-medium`) and the
Typesafe Jev decision model (`pratt-jev`). We want the LLM escalation backend
and Jev to use this router instead of OpenRouter.

**Decision.**
1. **Provider inference.** `load_config()` picks providers from whichever key is
   present, with explicit `*_URL` / `*_MODEL` overrides always winning:
   - LLM: `PRATT_LLM_API_KEY` → `https://llm.pratt.edu/v1/chat/completions`,
     `pratt-medium`; else `OPENROUTER_API_KEY` → OpenRouter chat completions,
     `openai/gpt-4o-mini`. `LLM_API_KEY` overrides the key.
   - Jev: `SYSTEMONE_API_KEY` → native TypeSafe (`/v1/systemone`,
     `jev-latest`); else `PRATT_LLM_API_KEY` → `https://llm.pratt.edu/v1/systemone`,
     `pratt-jev`; else `OPENROUTER_API_KEY` → `/api/alpha/decisions`,
     `typesafe/jev-1.13`.
2. The router's Jev surface is `POST /v1/systemone` (the OpenRouter mirror lives
   at `/api/alpha/decisions`). `pratt-jev` is a *decisions* model: it is
   **rejected** by `/v1/chat/completions` and `/v1/responses`
   ("decisions model … use the decisions endpoint"), so it must not be wired as
   a chat model.

**Why.** Keeping one inference rule (key ⇒ endpoint+model) means `.env` only
needs `PRATT_LLM_API_KEY`; the OpenRouter and native-TypeSafe paths remain intact
for portability, and every endpoint/model is still overridable.

**Consequences.** `.env.example` documents all three Jev options and the Pratt
LLM default. The disabled-Jev smoke test now also clears `PRATT_LLM_API_KEY`.
Live-verified 2026-10-11: `pratt-medium` extracted the sample (mentions +
relations); `pratt-jev` returned `score 1.84` (merge) for a near-duplicate ORG
pair and `noul 0.99` / `0.02` for a claim and its reversed form. Router usage is
reported per call (e.g. `cost: 1.47e-05`).

