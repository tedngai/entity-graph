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

---

## D13 — Relation candidates come from JointIE, not independent `.relations()`
**Status:** accepted · **Date:** 2026-10-11

**Context.** D8 decoded entities and relations in separate calls
(`create_schema().entities().relations()`). That path cannot enforce typed
endpoints, cardinality, or no-self-loops, so an organization can be emitted as
the founder of a person. GLiNER2.5's `JointIE` scores candidates and searches a
graph that already satisfies those constraints.

**Decision.** `GlinerExtractor` loads `JointIE` on `fastino/gliner2.5-base-v1`
(D8's checkpoint; `from_pretrained` accepts boundary models with relations
enabled). The schema lives in `extract/ontology.py` and is shared with the
local gate: closed entity labels, the existing relation set with typed
endpoints, `unique_head` on `leads` and `subsidiary_of`, `symmetric` on
`partnered_with`, and `no_self_loops()`. Decode with
`JointIEConfig(optimizer="beam", beam_size=32)`. `GLINER_THRESHOLD` is passed
as `entity_threshold` (an extra cut), not as the decoder's candidate floor.
If `result.feasible` is false, mentions are kept and relations are dropped —
an inconsistent graph is not stored. Independent `.relations()` remains only
when the installed `gliner2` has no `joint_ie` module.

**Why.** Typed endpoints are the constraint the independent decoder cannot
express. One ontology module keeps the extractor and the Decide relation
question from drifting apart. The pipeline already chunks and rebases offsets,
so extraction stays per window rather than calling `extract_long` inside the
extractor.

**Consequences.** `RelationProposal` gains optional `source_type` /
`target_type`. `relation_proposals` stores them (re-run `init-db`; migration
`db/migrations/001_gate_columns.sql`). D8's install notes still apply.

---

## D14 — Modality gate: Jev Choice beside the Noul; Decide when there is no key
**Status:** accepted · **Date:** 2026-10-11

**Context.** The assertion Noul asks whether the evidence states the claim. It
does not ask what kind of statement that is. A sentence that says a firm *will
invest* can score as a relation and still not be a completed fact. An
independent measurement (Lyon, 2026-09-24) found a local GLiNER2.5-Decide gate
within a hundredth of F1 of a hosted Jev assertion gate, using an argmax over
short modality labels plus a relation-agreement check. Tasks that share a
prompt shift each other's answers, so the two questions are separate passes.

**Decision.**
1. When a Jev key is set, `judge_relations` asks the existing assertion Noul
   and a `choice` question in the same request. Criteria are short:
   `fact`, `possibility`, `denial`, `forecast`, `not_stated`. Promotion
   requires modality argmax `fact` (no probability cutoff on that head) and
   Noul at or above the frozen cutoff (D15).
2. When no Jev key is set, `build` loads `fastino/GLiNER2.5-Decide` and asks
   the same two questions as separate `Classifier` calls. Relation agreement
   is asked only when both endpoint types are known; the options are the
   ontology's legal relations plus `no relation`. Disagreement drops the edge.
   Decide probabilities are not Nouls, so the 0.5 cutoff is not applied to them.
3. A claim neither judge can score is promoted on extractor confidence, matching
   the previous offline degrade. Jev wins when both could run; Decide is not
   loaded in that case.

**Why.** The two checks fail on different edges (right label, wrong modality
versus wrong label). Short labels are Decide's idiom; Jev's Choice already
uses a criteria map. Keeping both questions out of one Decide prompt follows
the measured result that a shared prompt costs recall.

**Consequences.** `relation_proposals.modality` and `.gate` record the judgment.
`DECIDE_MODEL` / `DECIDE_DEVICE` (default `GLINER_DEVICE`). No new threshold.

---

## D15 — Gold facts before any cutoff change; 0.5 stays frozen
**Status:** accepted · **Date:** 2026-10-11

**Context.** D9 set `RELATION_VALIDATION_THRESHOLD = 0.5` from one sample
(14 candidates, 9 promoted). That is not a precision/recall curve. Tuning it
before a gold set exists would fit the sample that produced the constant.

**Decision.** Do not change `0.5` in this change. Add
`data/gold/greengrid_partnership.jsonl`: asserted facts, relations the text
states in the wrong modality (`nonfact`), and the falsehoods the live run
already caught (`trap`, including reversed `FOUNDED` and fabricated
`ACQUIRED` / `SUBSIDIARY_OF`). `entity-graph eval` checks that every evidence
quote is a substring of the source, scores predictions (alias-aware, direction-
sensitive), and prints a threshold sweep. The sweep is informational. Writing
a new cutoff requires a new decision record after that report.

**Why.** P8.1 asked for a golden fact set before threshold tuning. The sample
is small; it is enough to stop the cutoff from moving on anecdote, not enough
to pick a new one.

**Consequences.** `entity-graph eval` needs no database. `--predictions` scores
a JSON/JSONL file; `--from-db` scores current facts when Postgres is up.

---

## D16 — OCR text is preprocessed at ingest; the document is its own de-hyphenation dictionary
**Status:** accepted · **Date:** 2026-10-11

**Context.** The Shanghai Capitalists book (P9.1) arrives as OCR output: an
`# OCR Results` preamble with a Windows source path, 334 `## Page N:
imageNNNNN.jpg` markers, 335 `---` rules, 668 code fences, 220
`\textsuperscript{N}` markers, Unicode superscript footnote digits, and
typographic punctuation. Two structural problems directly hurt extraction:
**262 of 334 pages end mid-sentence** (so page scaffolding cuts sentences and
entity names in half), and ~14 line/page breaks hyphenate a word (`Shang-\nhai`,
`Chi-\nnese`) — though some are genuine hyphenated romanizations (`Yueh-sheng`,
`Li-shih`, `profit-earning`) that must not be stripped.

**Decision.** `preprocess.py` cleans OCR text at **ingest**, before it is stored,
so mention offsets and evidence quotes index the same cleaned text (invariant
#1); the original file stays on disk and is recorded in `documents.source_uri`.
`clean_ocr_text`:
1. keeps only the fenced page bodies (drops the preamble, markers, rules,
   fences) and **rejoins pages** — a page ending mid-sentence continues on the
   next with a space, a page ending on terminal punctuation starts a new
   paragraph;
2. strips inline footnote markers (`\textsuperscript{...}`, Unicode
   superscripts) and unwraps `\textit{...}` (keeping its text);
3. **de-hyphenates using the document itself as its dictionary**: a break
   `head- tail` is joined only when `headtail` (case-folded) occurs elsewhere in
   the document, so attested compounds (`Yueh-sheng`, `profit-earning`) keep
   their hyphen;
4. normalises typographic quotes/dashes to ASCII and collapses whitespace.

Detection is conservative (`looks_like_ocr` requires an `# OCR Results`
heading or a `## Page N: <file>.<img>` marker), so ordinary Markdown is left
untouched. `PREPROCESS_OCR=false` disables it. `entity-graph preprocess
--input <file> --out <file>` materialises the cleaned text for inspection.

**Why.** Extraction, sentence splitting, and Jev evidence quotes all assume
prose; the scaffolding is pure noise and the page breaks are actively
destructive. Doing it at ingest keeps offsets consistent instead of adding a
second coordinate system. The self-referential de-hyphenation rule resolved
13/14 real cases (only the rare `count-less` is left hyphenated) without
needing a bundled dictionary.

**Consequences.** `documents.raw_text` now holds cleaned text for OCR inputs
(the checksum, hence idempotency, is over the cleaned text — invariant #5).
`config.preprocess_ocr` (env `PREPROCESS_OCR`, default on). Book run: 694,291 →
655,495 chars, zero scaffolding tokens remaining, all risky names preserved.

## D17 — Relation prompts are a selectable profile; the local Clef model is calibrated opt-in
**Status:** accepted · **Date:** 2026-10-11

**Context.** The local SystemOne-compatible server (Clef-Flash, D-none) is a
drop-in transport but, with Jev's wording, a poor judge: it promoted 5 of the 8
nonfact/trap gold items (gate accuracy 0.706 vs Jev 0.941). The failure was not
in the client — it was in the *questions we asked it*. A calibration sweep
(`scripts/calibrate_clef.py`) showed three effects: request batch size dominates
(0.824 at 12 claims/request → 0.941 at 1–2), modality criteria must name the
tense cues (`will/plans to/expansion into` → `forecast`), and the assertion must
name the doer/receiver with an explicit passive-voice warning to catch reversed
relations.

**Decision.** Relation-judgment wording and request shape live in
`extract/profiles.py` as a `RelationPromptProfile` (task, assertion instruction,
modality criteria, few-shot examples, batch size, optional provider→gate
modality map). `judge_relations(..., profile=...)` consumes one; `get_profile`
resolves a name; `SYSTEMONE_PROFILE` selects it (empty = the Jev-verified
default). Two profiles ship: `default` (unchanged Jev wording, 12/request) and
`clef` (doer/receiver assertion, tense cues, examples, 1/request). The profile
changes only wording — `RELATION_VALIDATION_THRESHOLD` stays frozen (D15) and is
applied identically to both providers.

**Why.** Calibration is prompt engineering against a fixed model, and it should
be data, not a fork of the client. Keeping it a named profile makes the change
auditable, testable offline, and reversible, and leaves Jev's verified wording
as the default so nothing regresses.

**Consequences.** `SYSTEMONE_PROFILE=clef` + the local server reaches gate
accuracy 0.941 with 0 leaks (promote-agreement with Jev 15/17) at ~2.5× the
wall time (sequential single-claim requests). It misses a different fact than
Jev, so it is a peer, not a superset. New profiles are additive; the default is
untouched. See `docs/RESEARCH.md` §4d-i.

## D18 — Non-node types are dropped, the graph view is connected-only, surfaces are normalized
**Status:** accepted · **Date:** 2026-10-11

**Context.** The Shanghai book run produced 4,191 entities for one book, but
**3,589 had no edges**: `DATE` 918/918 and `CONCEPT` 1,293/1,306 were ~100%
isolated, as were most `EVENT` (377/396) and `PLACE` (165/210). The meaningful
graph was ~602 connected entities / 990 edges. Three mechanisms inflated the
count: (a) extraction emitted `DATE` and `CONCEPT` mentions as entities; (b)
near-duplicate dates never merged — `1928`, `1928-1932`, `March 1928` each got
their own entity, partly because alignment routes such pairs to `curator`
(which *creates* a new entity); (c) `v_graph_nodes` exposed every entity, so
analysis and visualization saw 4,191 nodes.

**Decision.**
1. **Excluded ontology types** (`EXCLUDE_ENTITY_TYPES`, default `DATE`) are
   dropped at the extraction boundary: `pipeline.filter_extraction` removes
   mentions of an excluded type and any *typed* proposal with an excluded
   endpoint, and `build_from_proposals` refuses to promote a proposal whose
   endpoint type is excluded. Excluded surfaces still appear inside fact
   `evidence_text`, so provenance is intact. Storing dates as `facts.valid_from`
   is deferred to temporal analysis (P7.4).
2. **`v_graph_nodes` is connected-only**: an entity with no incident
   relationship is hidden from the view. Entity rows are never deleted
   (invariant #2); only the contract view filters them (migration 002).
3. **Surface normalization** before resolution: `normalize_surface` collapses
   whitespace and trims edge punctuation; `match_surface` additionally reduces a
   `DATE` surface to its first year, so ranges and month variants share a node
   (the original surface is kept as an alias).

**Why.** An edgeless node carries no network signal, and dates are attributes of
events, not actors. Dropping them at the boundary — rather than post-hoc —
means they never pollute mention counts, resolution calls, or the graph.
Normalization cuts duplicate entity creation without a schema change.

**Consequences.** Book run: 4,191 entity rows → **602 connected nodes**, edges
unchanged at 990. `EXCLUDE_ENTITY_TYPES=` restores the old extraction behavior;
`DATE,CONCEPT` drops concepts too. The view change is a *contract* change
(invariant #4): code still reads through `v_graph_nodes`/`v_graph_edges`, which
now hide isolated entities. Existing databases pick up the view via
`entity-graph init-db`; the extraction and normalization changes apply on the
next extraction/resolution run.

## D19 — Curation is two reversible primitives: merge (repoints edges) and hide
**Status:** accepted · **Date:** 2026-10-11

**Context.** The Shanghai book graph carried two kinds of noise. (a)
**Duplicates** — `Soong` and `T. V. SOONG` (the latter mis-typed `CONCEPT`) are
one person; so are `Tu` / `Tu Yueh-sheng`, `Lamont` / `Thomas W. Lamont`, and
`Addis` / `Sir Charles Addis`. (b) **Non-actors** — generic concepts (`central
bank`, `finance`, `banking`, `trust`, `GOVERNMENT`) and bibliography references
(author/press pairs like `Fewsmith`–`University of Chicago`) that have edges but
are not part of the network the book is about. There was no way to clean either
without violating invariant #2 ("never hard-delete entities or facts").

**Decision.** Add three reversible primitives (`src/entity_graph/curate.py`):
1. **`rename(entity, new_name)`** changes the canonical name and keeps the old
   one as an alias (e.g. `Soong` → `T. V. Soong`).
2. **`merge(winner, loser)`** repoints the loser's relationships to the winner —
   collapsing any that become duplicates (facts move to the surviving edge) —
   moves aliases and mentions over, and sets `merged_into`. The loser row stays.
   Repointing is *required*: `v_graph_edges` joins on `merged_into IS NULL` for
   **both** endpoints, so marking a merge without repointing would silently drop
   every edge incident to the loser.
3. **`entities.is_hidden`** removes an entity *and its edges* from the graph
   views while keeping the row and its facts. Used for non-actors.
   `entity-graph curate --file curation.json` applies a plan
   (`{"renames": [[old, new]], "merges": [[winner, loser]], "hide": [...],
   "unhide": [...]}`); `--dry-run` previews it. No name in the plan is ever
   deleted.

**Why.** A merge capability was genuinely missing — the curator queue could flag
uncertain pairs but nothing could resolve them. Hiding is the minimal mechanism
for "persisted for provenance, but not a node", parallel to excluded types
(D18). Both are reversible (set `merged_into`/`is_hidden` back), so curation is
never destructive.

**Consequences.** Migration 003 adds `is_hidden` and redefines
`v_graph_nodes`/`v_graph_edges` to exclude hidden entities; existing databases
pick it up via `entity-graph init-db`. The graph contract now hides merged,
hidden, and isolated entities. Book-specific curation lives in
`network_analysis/shanghai/` (not in this repo). Self-loops created by a merge
are kept when they carry facts and dropped when they are childless.

---

## D20 — The Pratt router's default Jev model is `pratt-d3`
**Status:** accepted · **Date:** 2026-11-01

**Context.** D12 wired the Pratt branch of `load_config()` to `pratt-jev`. The
router now also serves `pratt-d3` on the same `POST /v1/systemone` decisions
surface, and the owner directed standardizing on it; the current experiment
(`network_analysis/anthropic-s1/`, the Anthropic PBC S-1) runs its gate on
`pratt-d3`.

**Decision.** When Jev credentials are inferred from `PRATT_LLM_API_KEY` (no
explicit `SYSTEMONE_URL` / `SYSTEMONE_MODEL` / `SYSTEMONE_API_KEY`), the model
is `pratt-d3` — one constant in `load_config()`, mirrored in
`compare_providers.py`'s `--a-model` default. Explicit `SYSTEMONE_*`
overrides still win, so a workspace can pin `pratt-jev`, native TypeSafe, or
the local Clef-Flash server (plus the `clef` profile, D17) exactly as before.
The default prompt profile stays the Jev-verified wording: `pratt-d3` is a
hosted router model, not the small local model the `clef` profile was
calibrated for.

**Why.** One obvious default keeps `.env` minimal (key only) while every
workspace keeps full control through environment overrides — the same
mechanism the per-document experiment folders already use for database
isolation.

**Consequences.** The frozen `RELATION_VALIDATION_THRESHOLD` (D15) carries
over unchanged; `pratt-d3` has not been through a gold-set sweep yet, so
re-run `entity-graph eval` before trusting it for threshold work.
Smoke-tested live 2026-11-01 (`/v1/systemone`, `pratt-d3`): noul 0.986,
choice 0.997 on a registration-statement probe.




## D21 — Symmetric relations are decoded directed and canonicalized afterwards
**Status:** accepted · **Date:** 2026-11-01

**Context.** The S-1 evaluation found zero `PARTNERED_WITH` proposals across
1,421 (S-1) and 2,520 (Shanghai book) proposals, although the relation is in
the ontology. Probing JointIE directly showed why: with `symmetric=True` the
decoder either never emits the relation or the greedy repair fails and the
window comes back *empty* — entities included. "Anthropic partnered with
Google and Broadcom" returned nothing; without the flag it returned
`partnered_with` at 0.97/0.95. Infeasible windows were 118/3,566 on the S-1
and 79/1,272 on the book, and each lost all of its mentions, contrary to the
D13 promise that an infeasible window keeps its mentions.

**Decision.** `relation_kwargs` no longer passes `symmetric` to JointIE.
`RelationSpec.symmetric` stays as ontology metadata: `parse_joint_payload`
swaps a symmetric pair into case-insensitive lexical order
(`canonical_pair`), and `build` orders the endpoint ids and stores the edge
with `is_directed = false`, so A~B and B~A are one edge. An infeasible window
is re-decoded with an entities-only schema, so it keeps its mentions; its
relations are still dropped (D13).

**Why.** Measured on 446 evenly spaced S-1 windows: infeasible 18 → 0,
entities 4,716 → 4,988, `partnered_with` 0 → 149 candidates; the other
relations are unchanged or slightly up. The new candidates include noise
(`Claude ~ customers`), which is what the relation gate is for.

**Consequences.** Existing databases keep their old proposals; re-extract to
gain partnerships. More ORG–ORG candidates means more gate calls at build.

## D22 — A proposal Jev did not answer stays pending
**Status:** accepted · **Date:** 2026-11-01

**Context.** `should_promote(None)` promotes an unscored proposal — the
offline degrade path when no gate exists (D14). But with Jev configured, a
missing verdict means the call failed. On the S-1, 17 proposals timed out and
were promoted on extractor confidence; 5 survived curation and 3 of those
were wrong (`Anthropic ACQUIRED Google` and `ACQUIRED Broadcom`, both from an
exhibit-index table row, and a signature-block self-loop), against 14 wrong
out of 214 judged edges.

**Decision.** `awaits_verdict(judgment, remote_gate=...)`: when the Jev client
is enabled and a proposal has no verdict, `build` leaves it `pending`, logs
the count, and the next `build` retries it. With no key (Decide or nothing)
and with `--no-validate`, behaviour is unchanged.

**Consequences.** A persistently failing endpoint leaves proposals pending
rather than silently promoting them; `build --no-validate` remains the
explicit way to promote without a gate.

## D23 — The graph views never expose a self-loop
**Status:** accepted · **Date:** 2026-11-01

**Context.** `build` skips self-relations and JointIE forbids them, but a
curation merge can join an edge's two endpoints. `_merge` keeps such a row
when it has facts (invariant #2), so it reached the graph: on the S-1,
`ANTHROPIC SUBSIDIARY_OF PBC` (a signature block) became
`Anthropic SUBSIDIARY_OF Anthropic` once `PBC` was merged into `Anthropic`.

**Decision.** Migration 004 (mirrored in `db/schema.sql`): `v_graph_edges`
filters `source_id <> target_id`, and `v_graph_nodes` counts only non-loop
edges when deciding whether an entity is connected. The relationship row and
its facts are untouched. `tests/test_views.py` checks the final view text.

**Consequences.** Applied to `entity_graph_anthropic_s1`: 219 → 218 edges,
nodes unchanged. Unmerging still does not restore the pre-merge endpoint of a
collapsed edge (unchanged D19 limitation).

## D24 — Defined terms resolve to their long form, scoped to their document
**Status:** accepted · **Date:** 2026-11-01

**Context.** Documents declare short names inline — `Long-Term Benefit Trust
("LTBT")`, `Anthropic, PBC (the "Company")`, `Clinton Health Access
Initiative (CHAI)`. Trigram and Jev alignment cannot see that an acronym and
its expansion are one entity, so the S-1 kept `LTBT` and `Long-Term Benefit
Trust` as separate nodes, and CNAS / USG needed manual curation.

**Decision.** `defined_terms.py` finds quoted parentheticals (with optional
`the` / `each, a` / `together with its affiliates,` prefixes) and bare
parenthesized acronyms, and anchors each to the extracted mention ending just
before it. It accepts a definition only when the short form can name that
mention: an acronym must spell its initials (minor words and corporate
suffixes skippable, hyphenated words split), a word term must consist of the
long form's own words, or be a conventional party name (`Company`, `Trust`,
`Registrant`, ...) for an organization. Pronouns never count. During
`resolve`, a mention whose surface is a defined short form of its document
resolves as the long form. Nothing is written to `entity_aliases`: a
definition binds only inside its own document.

**Why.** On the S-1 the unguarded pattern found 98 definitions, about a third
wrong — the quoted term often names something new right after an unrelated
noun (`Bartz, et al. v. Anthropic PBC ("Bartz")`, `... in the United States
("GAAP")`). With the guard, 59 definitions, all correct on review (LTBT, AWS,
GCP, CNAS, CHAI, USG, Company → Anthropic, PBC, Board → board of directors).
The Shanghai book yields one (B.A.T.); its quoted parentheticals are
translations of titles, which have no preceding mention match.

**Consequences.** Applies to new `resolve` runs; existing databases are
unchanged until re-resolved. A true alias that is neither an acronym nor a
word subset (`Google` ← `Alphabet Inc.`, `Bun` ← `Codeblog Corp.`) is missed
on purpose.

## D25 — Facts carry validity dates read from their evidence
**Status:** accepted · **Date:** 2026-11-01

**Context.** DATE is not a node type (D18), so no date reached the graph and
none of the S-1's 490 facts had `valid_from` / `valid_to`, although its
biographies date every role.

**Decision.** `temporal.fact_validity(evidence, objects, subjects)` reads a
date only from prepositional expressions (`from X to Y`, `between X and Y`,
`since X`, `until X`, `in X`, `as of X`, `1928-1932`) in the sentence naming
the fact's object: the first after the object, else the last before it —
unless another organization (`at Google Brain`) or a new clause
(`, and as a Director of`) sits between. Month precision is kept; a bare year
starts January 1 and ends December 31. Sentence splitting here does not break
after honorifics, initials, or `Inc.`. `build` dates every new fact;
`entity-graph dates` backfills facts whose dates are both NULL.

**Why.** Applied to `entity_graph_anthropic_s1`: 63 facts dated. The bio
ranges come out exactly (Rao: Bain 2005-09..2008-07, Blackstone, Airbnb,
Cedar, Fanatics; Daniela Amodei at Stripe 2013-05..2018-07; Liddell in the
U.S. government 2017-01..2021-01; acquisitions dated to their month). Without
the blocker rule, "Before joining OpenAI, Dr. Amodei served ... at Google
Brain ... from October 2015 to June 2016" dated the OpenAI fact.

**Consequences.** A heuristic, not a parser: an event date in the same
sentence can be read as a tenure date (Fontaine's CNAS membership got the
August 2026 donation date). Analysis that depends on chronology should check
the evidence.

## D26 — One pipeline, a document profile per kind of document
**Status:** accepted · **Date:** 2026-11-01

**Context.** The S-1 evaluation split its failures in two. Some were bugs
that hurt every document (D21–D25). The rest were about the kind of
document: an SEC filing needs litigation, supplier, underwriter, auditor and
counsel relations and a product type (Claude was typed PERSON); its exhibit
index and signature pages produced the worst edges; its generic role nouns
differ from a history book's. Separate pipelines would duplicate resolution,
the gate, the fact store, curation and analysis — the domain-neutral parts
and the strongest ones.

**Decision.** `doc_profiles.py` defines `DocumentProfile`: an `Ontology`
(entity descriptions, label → stored type, relation specs), regexes for
sentence windows not to mine, defaults for the gate profile and excluded
types, default curation hides, and a gold file. `DOC_PROFILE` selects one
(`general`, `sec_filing`, `history_book`); explicit env vars still override
its defaults. The extractor builds its JointIE schema from the profile, the
Decide gate asks agreement over the profile's relations, `extract` skips
boilerplate windows, `curate --defaults` applies the profile's hides,
`eval` defaults to the profile's gold file, and `entity-graph profiles` lists
them. Profiles extend the general relation set rather than replace it.
`data/gold/anthropic_s1.jsonl` is the first non-sample key: 60 facts across
governance, capital, commercial and litigation, plus 7 traps taken from the
S-1 run's wrong edges; symmetric relations match either direction.

**Why.** Measured on S-1 windows with `sec_filing`: 17 of 3,566 windows
skipped (a one-link rule would have skipped 514, since every page repeats a
"Table of Contents" link); `SUED`, `UNDERWRITER_OF`, `ADVISOR_TO` and
`DEVELOPS` proposals now appear (e.g. `Anthropic PBC SUED Department of
War`). `eval --from-db` on the pre-change S-1 database: recall 39/60, all 7
traps leaked — the baseline to beat.

**Consequences.** New relations bring noisy candidates (`we UNDERWRITER_OF
RSUs`) that only the gate removes, so a profile is not trusted until its gold
eval is run end to end. Product typing improves but is not settled: in the
sample, `Claude` was PRODUCT 15, PERSON 21, ORG 10 times; an entity keeps the
first type it is created with (fixed by D27). `history_book` has no gold file yet.


## D27 — An entity's type is the majority of its mentions' types
**Status:** accepted · **Date:** 2026-11-01

**Context.** An entity kept the type of the mention that created it. In the
S-1 re-run (P9.3) `Claude` had PRODUCT 214, ORG 123 and PERSON 109 mentions
but was stored PERSON; `Anthropic` had ORG 105 and PRODUCT 5 but was stored
PRODUCT, so alignment compared it as a product and it never joined
`Anthropic, PBC`. Build checked a relation only against the proposing
mention's type, so the 3 of 22 `Claude WORKS_FOR` proposals typed PERSON
passed and leaked the last gold trap.

**Decision.** `entity_types.py`: `vote_type` takes the plurality of an
entity's mention types (MISC and empty vote only when nothing else does; a
tie keeps the current type, else breaks alphabetically). `resolve` re-votes
the entity after every mention it links, so later Jev alignment calls see the
majority type. `build` rejects a proposal whose relation the endpoint
entities' types do not allow (gate `entity_type`); unknown or MISC types never
block. `entity-graph retype [--retire-facts] [--dry-run]` re-votes an existing
database and lists, or marks not current, the facts the new types forbid.

**Why.** On `entity_graph_anthropic_s1_v2` (curated): 176 entities change
type; 117 of 773 current facts are type-illegal under the voted types. A
sample of 45 was 44 noise (`prospectus LOCATED_IN Dubai`, `Claude WORKS_FOR
insurance`, `Dario Amodei FOUNDED NEOs`). Retiring them: recall stays 40/60,
trap leaks 1/7 -> 0/7, current facts 773 -> 656.

**Consequences.** A wrong majority now overrides a right mention: generic
nouns retype freely (`customers` ORG -> PERSON) and an entity whose mentions
split across types loses the edges of its minority reading (`Trustees` ORG
lost `WORKS_FOR Long-Term Benefit Trust`). Re-voting during resolve cannot
undo an alignment decision taken before the vote turned, so the Anthropic /
`Anthropic, PBC` split on the existing S-1 database needs a fresh resolve (or
a curation merge) to disappear. Retired facts stay in `facts` with
`is_current = false`.

## D28 — A document's "we" is extracted as the entity it defines
**Status:** accepted · **Date:** 2026-11-01

**Context.** A held-out S-1 (CoreWeave, P9.6) scored 23/54 against 40/60 on
the Anthropic S-1 the `sec_filing` profile was built on. Nearly every miss had
the registrant on one side and died at extraction: the filing says "Nitin
Agrawal has served as our Chief Financial Officer", and GLiNER2.5 tagged 543
of about 6,500 "our"/"we"/"us" as an organization, so no relation was ever
proposed. Hand-curated merges (we/us/company → registrant) fix none of this:
they cannot merge a proposal that does not exist.

**Decision.** `self_reference.py` reads the definition corporate documents
give their author (`the terms "CoreWeave," the "Company," "we," "us," and
"our" refer to CoreWeave, Inc.`). When a document has one, each window is
extracted twice: as written, and with the defined terms replaced by the
author's name ("our company" → the name, "our" → "Name's", "we"/"us" → the
name, "the Company" when defined). Mentions and evidence of the rewritten
reading are mapped back to the original offsets and text, so provenance stays
verbatim; the two readings are unioned. Resolution maps the defined terms to
the author through the D24 term map, so "we", "our" and "the Company" become
one entity with no curation. A document without such a definition is
untouched (in a memoir "we" is the writer, not a party). `entity-graph probe`
measures proposal recall on the windows holding gold evidence in minutes,
with `--gate` to send the matching proposals to the relation gate.

**Why.** Probe proposal recall, no curation (greengrid / Anthropic / CoreWeave):
7/9, 36/60, 22/54 before; rewrite only 7/9, 31/60, 34/54 ("our audit
committee" read as "Anthropic's audit committee" pulls member edges onto the
company); union 7/9, 42/60, 39/54; plus "our company" → name 7/9, 45/60,
39/54. pratt-d3 keeps 48/49 and 38/40 of the matched true proposals with the
evidence in the document's own words; telling it who "we" is (a claim
`context` note) changed nothing (47/49, 38/40) and was dropped.

**Consequences.** Windows with a defined term are extracted twice (about
twice the extraction time on a filing). More traps are proposed (Anthropic
2, CoreWeave 2 in the probe); the gate and the D27 type check must reject
them. A testimonial quoted in the document ("transformative for our
infrastructure") is read as the author's.

## D29 — The relation label the model reads can differ from the stored name
**Status:** accepted · **Date:** 2026-11-01

**Context.** Two S-1s name ten suppliers; `SUPPLIES` was proposed 1 and 9
times. On the sentence "Anthropic purchases compute capacity from Fluidstack,
Lambda and Nscale, and GPUs from AMD" JointIE emitted `partnered_with` and
`subsidiary_of`, never `supplies`. JointIE accepts a relation description but
ignores it: output was identical with and without one.

**Decision.** `ontology.RELATION_LABELS` maps a stored relation name to the
phrase JointIE is asked for; the extractor maps the answer back to the stored
name. Only `supplies` → "sells to" (same head → tail direction) is set.

**Why.** On five supplier/customer sentences "sells to" found 4 of 5 pairs in
the right direction ("supplier of" 2, "provides services to" 4 with noise).
Probe: Anthropic 45/60 → 49/60 (AMD, Fluidstack, Lambda, Nscale), CoreWeave
39/54 → 40/54, greengrid unchanged. Writing every label in plain words
("works for") lost one fact; "underwrites" / "audits" / "legal counsel to"
lost one; neither was kept.

**Consequences.** A label is chosen by measurement, not wording taste; add
one only with a probe result on every gold set.

## D30 — A relation to one member of a list is proposed for every member
**Status:** accepted · **Date:** 2026-11-01

**Context.** Lists of parties ("Morgan Stanley & Co. LLC, J.P. Morgan
Securities LLC, and Goldman Sachs & Co. LLC are acting as representatives")
come out of JointIE with one or two members linked.

**Decision.** `coordination.py`: after a window is extracted, same-type
mentions joined only by commas, "and" or "or" form a list; a relation with an
endpoint in a list is copied to the other members (confidence × 0.9, same
evidence). A relation between two members of one list is not copied. The gate
judges every copy.

**Why.** Probe, no curation: CoreWeave 40/54 → 42/54 (Morgan Stanley and J.P.
Morgan UNDERWRITER_OF CoreWeave); Anthropic 49/60 and greengrid 7/9
unchanged. One more trap proposed (Razavi MEMBER_OF Class T), which the gate
rejected in the earlier full run.

**Consequences.** Lists whose members get no relation at all (the four LTBT
trustees, "Switch and Chirisa Technology Parks") are not helped.

## D31 — Evidence takes the next sentence when it names an argument again
**Status:** accepted · **Date:** 2026-11-02

**Context.** JointIE anchors each argument on its first mention, so the
evidence quote (head sentence through tail sentence) can stop one sentence
short of the claim: "Our compensation committee is composed of Karen Boone and
Glenn Hutchins. Mr. Hutchins is the chair of our compensation committee." The
gate saw only the first sentence and rightly rejected Hutchins LEADS.

**Decision.** `_evidence_for` extends the quote by the following sentence when
that sentence contains the head or tail surface again (case-insensitive). One
sentence at most; no pronoun resolution.

**Why.** The full uncurated CoreWeave run (old quotes) rejected both Hutchins
LEADS committee facts (0.07, 0.15). Gate probe with D31, no curation: CoreWeave
facts kept 41/42 proposed; Anthropic 48/49 and greengrid 6/7 unchanged; no new
trap kept. (Anthropic's
third kept trap, Razavi MEMBER_OF Class T, was proposed since D30 and is
independent of this change.)

**Consequences.** Some quotes are one sentence longer, so the gate prompt grows
slightly.

## D32 — Keep up to three evidence quotes per relation across windows
**Status:** accepted · **Date:** 2026-11-02

**Context.** `extract_document` collapsed proposals with the same (source,
target, relation) across windows to the single most confident one. The
extractor's confidence is about the pair, not about the quote: in the uncurated
Anthropic run, Goldman Sachs and Morgan Stanley UNDERWRITER_OF Anthropic kept a
lock-up paragraph as evidence and the gate rejected it (0.32, 0.27); the window
saying they "are acting as representatives of the underwriters" was dropped
before the gate saw it.

**Decision.** Keep up to `EVIDENCE_PER_RELATION = 3` distinct evidence quotes
per key, most confident first. Build already judges and promotes each proposal
on its own, so a key is promoted when any of its quotes passes.

**Why.** The probe extracts each gold window alone and cannot show this;
measured on full uncurated runs (P9.7). With D31, uncurated end to end:
Anthropic 48→50/60 (`entity_graph_anthropic_s1_v5`, 1 trap leak, unchanged),
CoreWeave 44→46/54 (`entity_graph_coreweave_s1_v4`, 0 leaks). Promoted edges
1,617 and 1,110.

**Consequences.** More gate calls for relations repeated across windows (at
most 3× per key).

## D33 — An LLM reads relations among the entities GLiNER found (opt-in)
**Status:** accepted (opt-in, `RELATION_READER=all`) · **Date:** 2026-11-02

**Context.** After D28–D32 the remaining misses are relations GLiNER never
links although it tags both entities: auditor, legal counsel, acquisitions,
lawsuits, list members with no verb of their own. D11's escalation triggers
(no mentions, no relations, sparse structure) are per document and never fire
on a filing.

**Decision.** `extract/relation_reader.py`: per window, after GLiNER, an LLM
gets the window, the entity strings GLiNER found, who "we" is (D28) and the
profile's relations with a one-line head -> tail meaning, and returns the
relations the text states among those entities. Both ends must be listed
entities, the relation must be in the profile and legal for their types, and
the evidence must be a verbatim quote (else the sentences spanning both ends).
Proposals join GLiNER's before coordination (D30); the gate judges them.
Model: `LLM_MODEL` (Pratt default `pratt-medium`, an internal deployment) with
`reasoning_effort: low`. Every LLM call is capped at 2,048 reply tokens.

**Why.** Probe with the gate, no curation (proposed / kept / traps kept):
greengrid 7→9 / 6→8 / 0; Anthropic 49→56 / 48→55 / 3→2; CoreWeave 42→51 /
41→51 / 0. 114 calls, 96k prompt + 25k reply tokens (~1,060 per window), no
failures. Uncapped, `pratt-medium` reasoned 32,768 tokens on one window and
answered nothing; with thinking disabled it missed Fenwick counsel.

Full uncurated runs, D31+D32 without → with the reader: Anthropic 50→55/60
(`entity_graph_anthropic_s1_v6`; trap leaks 1→2), CoreWeave 46→51/54
(`entity_graph_coreweave_s1_v5`; 0 leaks). Promoted edges 1,617→2,106 and
1,110→1,485 (+30–34%). Reader cost: Anthropic 3,511 calls, 2.83M prompt +
0.21M reply tokens, 2 failed; CoreWeave 1,762 calls, 1.58M + 0.11M, 0 failed.
Still missed: three LTBT trustees and Nscale SUPPLIES (Anthropic), Pure and
Vast PARTNERED_WITH (CoreWeave). Extraction with the reader took ~2h20m per
filing (4 workers, both filings at once) against ~2h without.

Precision audit (P9.7), promoted edges outside gold, 30-edge seeded samples
per stratum per filing, judged by hand against the stored evidence (correct =
the quote states that relation between two named entities, right direction):
edges in both runs (GLiNER's) 9/60 correct, +2 borderline (CoreWeave 6/30,
Anthropic 3/30); reader proposals 30/40 (18/22, 12/18); edges that appear in
one run only and did not come from the reader (run-to-run churn) 1/20. Of the
GLiNER errors 34/60 have a generic endpoint ("employees", "the company",
"underwriters", "chief executive officer") and 15 a wrong relation or one the
text does not state (director as ADVISOR_TO or WORKS_FOR, lender as
PARTNERED_WITH). So the reader raises precision as well as recall; the
graph's precision problem is generic GLiNER endpoints the gate does not
judge. About half of every run's edges have an all-lowercase endpoint, but a
filter on that alone would drop 12–16 gold edges (the registrant's own "board
of directors" and committees), so it needs an exemption for those.
Reader proposals were stored with extractor `gliner2` until
`RelationProposal.extractor` was added; they are now `llm_reader`.

**Consequences.** Entity-count triggers select 85–95% of filing windows (they
are entity-dense), so a full S-1 costs ~3.8M tokens (Anthropic) / ~1.9M
(CoreWeave) at one call per window; about two thirds of each prompt is fixed
overhead (entity and relation lists). Gate claims on gold windows rose
~1.5–1.8×. Precision outside the gold sets is unmeasured. Off by default.

## D34 — An LLM-only extraction backend (`llm_ie`), trialled against GLiNER and the hybrid

**Context.** The D33 audit found GLiNER's promoted edges outside gold ~15%
correct (mostly generic endpoints) and the reader's 75%, but the reader only
links entities GLiNER found, so an LLM had never been tested on its own.

**Decision.** `EXTRACTION_BACKEND=llm_ie` (`extract/llm_ie.py`): one call per
window (`pratt-medium`, low reasoning) returns entities and relations. It gets
the profile's entity types and relations, who "we" is, and a rule: specific
named entities only, no generic nouns or roles, except the registrant's own
board and committees. Entities must be found verbatim (offsets kept);
relations pass the reader's checks; facts carry extractor `llm_ie`. Windows
run in the pipeline's 4-worker pool; GLiNER is not loaded. Gate calls now wait
`SYSTEMONE_TIMEOUT` (default 120 s): the router queues requests under load,
and at 20 s the D33 Anthropic build dropped 261 queued batches.

**Why (trial, uncurated, same windows and gate).** Gold-window probe, proposed
/ kept: greengrid 8/7, Anthropic 51/50, CoreWeave 50/49; traps proposed 0
(hybrid 5); gate claims less than half the hybrid's. Full runs
(`entity_graph_coreweave_s1_v6`, `entity_graph_anthropic_s1_v7`):

| | GLiNER | Hybrid (D33) | LLM-only |
|---|---|---|---|
| CoreWeave gold | 46/54 | 51/54 | 51/54 |
| Anthropic gold | 50/60 | 55/60 | 49/60 |
| Promoted edges (CW / An) | 612 / 858 | 753 / 1,024 | 299 / 315 |
| Non-gold edges correct, strict (lenient) | ~15% | ~15% + reader 75% | CW 68% (83%), An 68% (95%) |
| LLM tokens (both filings) | 0 | 4.4M | 5.5M |
| Wall time per filing | ~3.5 h | ~5.5 h | 20–30 min |

Audit: 40 seeded random non-gold edges per filing, judged by hand; strict =
relation stated between two named entities, right direction; lenient adds
defensible readings (lender as SUPPLIES, trustee or manager as MEMBER_OF).
Generic endpoints: 3 of 80 (GLiNER 34 of 60). `trap_leaks` in `eval` counts
facts, so one trap edge with three quotes (D32) reports 3; the only trap edge
that leaks in any design is LTBT SUBSIDIARY_OF Anthropic.

Anthropic misses: the three other LTBT trustees (missed by every design), the
four Public Benefit Fund managers proposed as MEMBER_OF where gold has LEADS,
Amazon INVESTED_IN promoted under "Amazon Web Services" (resolution merged
Amazon into AWS), Google INVESTED_IN under "Google LLC" (not a gold alias; the
hybrid misses it the same way), SpaceXAI SUPPLIES and Bartz SUED (gate 0.42,
0.48).

**Consequences.** The rule fixed before the trial (recall within 2 of the
hybrid on both filings and ≥75% precision) is not met: Anthropic is 6 behind
and strict precision is 68%. The hybrid stays the recommended configuration.
The LLM-only graph is far cleaner (whole-graph precision roughly 75–95%
against ~25–35% for the hybrid, whose GLiNER edges the rule did not weigh)
and 10× faster at similar token cost; the Anthropic gap is concentrated in
relation choice for managers and one resolution merge. Off by default.

**Follow-up: larger models, reasoning off.** `pratt-high` and `pratt-agent` at
`reasoning_effort` low reasoned to the reply cap (2,048) and answered nothing;
at 8,192 one window cost 54 s / 7.4k tokens (high) and 11.5 s / 3k (agent).
`LLM_REASONING_EFFORT=none` turns reasoning off on both (~3 s, ~1.5k tokens a
window); `low` stays the default for `pratt-medium`. Full runs, same prompt,
gate and audit (`coreweave_s1_v7`/`v8`, `anthropic_s1_v8`/`v9`):

| | pratt-medium (low) | pratt-high (none) | pratt-agent (none) |
|---|---|---|---|
| CoreWeave gold | 51/54 | 49/54 | 45/54 |
| Anthropic gold | 49/60 | 49/60 | 44/60 |
| Promoted edges (CW / An) | 299 / 315 | 275 / 323 | 217 / 248 |
| Non-gold correct, strict (lenient) | 68% / 68% (83% / 95%) | 72.5% / 72.5% (97.5% / 92.5%) | 82.5% / 75% (95% / 92.5%) |
| Trap leaks | LTBT SUBSIDIARY_OF | none | none |
| LLM tokens (both filings) | 5.5M | 5.4M | 5.6M |
| Extract time per filing | 20–30 min | 8–14 min | 6–9 min |

`pratt-high` finds every CoreWeave governance fact (36/36) and Amazon
INVESTED_IN on Anthropic, but misses CoreWeave's data-center suppliers (Core
Scientific, Switch, Chirisa read as partners) and NVIDIA. `pratt-agent` is
the most precise and the least complete: on Anthropic it proposes none of the
nominating committee, Dr. Shah's LTBT seat or two underwriters. Neither meets
the rule (high: Anthropic 6 behind, 72.5% strict; agent: 6 and 11 behind), so
the hybrid stays the default. Errors left in all three are shared: managers
and trustees as MEMBER_OF/ADVISOR_TO, directors as WORKS_FOR from the
management table, the LTBT trustees and Public Benefit Fund managers.
