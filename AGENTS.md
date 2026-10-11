# AGENTS.md — Operating Manual for AI Agents

Working rules for any AI agent (or human) doing work in this repository.
Before starting a task, read `PROGRESS.xml` to see where things stand, and
update it when you finish.

**Commit and push as you go.** A finished slice is committed and pushed to
`origin` before the next slice starts. Do not save commits or pushes for the
end of a task. See §6.

## 1. Project mission

Build a system that **extracts entities and their relationships from
unstructured documents**, stores **key facts with full provenance** in
PostgreSQL, and uses **NetworkX** to analyze and visualize the resulting
relationship network.

Non-goals (for v1): a Neo4j/JanusGraph deployment, RDF/SPARQL, real-time
streaming ingestion, multi-tenant isolation.

## 2. The stack (do not change without a decision record)

- **Python 3.11+** (`py -3.12` on this machine), package `entity_graph` under `src/`
- **PostgreSQL 16** via `docker-compose.yml` (image `pgvector/pgvector:pg16`)
- **Schema**: `db/schema.sql` is the single source of truth. Migrations are
  new numbered files in `db/migrations/` once the first migration exists.
- **Extraction**: pluggable backends implementing `entity_graph.extract.base.Extractor`
  (`baseline` → `spacy` → `gliner2` → `llm`). Primary path is GLiNER2.5
  **JointIE** (`extract/ontology.py` + `extract/gliner_extractor.py`, D13).
  spaCy is a lightweight fallback; the LLM backend is an **opt-in
  escalation wrapper** (`EscalatingExtractor`) over a cheap backend, never the
  default (D2, D11).
- **Ingestion**: OCR documents are cleaned at ingest (`preprocess.py`, D16) —
  page scaffolding, footnote markers, page-break rejoin, and de-hyphenation —
  before storage, so offsets/evidence index the cleaned text.
- **Ontology types that are not nodes** are dropped at extraction
  (`EXCLUDE_ENTITY_TYPES`, default `DATE`, D18): their mentions and typed
  proposals never reach the database, and `build` refuses to promote a proposal
  with an excluded endpoint type. Their surface still appears in fact evidence.
- **Jev / System One (TypeSafe)**: typed decision calls for entity alignment,
  type arbitration, and relation validation — see `src/entity_graph/extract/jev.py`
  and `docs/RESEARCH.md` §4. It returns `Choice`/`Score`/`Noul` decisions, it is
  **not** an NER model and never generates spans. Relation judgment is a Noul
  plus a modality Choice (D14), and its wording is a selectable profile
  (`extract/profiles.py`, D17): `SYSTEMONE_PROFILE=clef` selects the calibrated
  local-model wording (doer/receiver assertion, tense cues, one claim per
  request) instead of the Jev-verified default. With no key, `extract/decide.py`
  is the local gate. Do not tune `RELATION_VALIDATION_THRESHOLD` without a
  gold-set report and a new decision record (D15).
- **Analytics**: NetworkX (`MultiDiGraph`), loaded only through
  `graph_store.load_graph()` which reads `v_graph_nodes` / `v_graph_edges`.
  `v_graph_nodes` is **connected-only** — an entity with no incident
  relationship is hidden (D18). Entity rows are never deleted; only the view
  filters them.

## 3. Commands (run from repo root)

```powershell
docker compose up -d                      # start Postgres
pip install -e .[dev]                     # install (in .venv)
entity-graph init-db                      # apply db/schema.sql
entity-graph ingest --input data/samples  # load documents (OCR text is cleaned first)
entity-graph preprocess --input f.md --out f.clean.md  # clean OCR text only (no DB)
entity-graph extract                      # run extraction backend
entity-graph resolve                      # entity resolution (+ Jev if API key set)
entity-graph curator                      # list uncertain pairs queued by Jev alignment
entity-graph run --input data/samples     # full pipeline
entity-graph analyze                      # NetworkX report to stdout
entity-graph visualize                    # out/graph.html (pyvis)
entity-graph eval                         # gold-fact report; does not tune thresholds
py -m pytest                              # tests (no DB or network needed)
```

## 4. Data rules (invariants — violating these is a bug)

1. **Provenance is mandatory.** Every mention carries document + char offsets;
   every fact carries document + evidence quote + extractor name.
2. **Never hard-delete entities or facts.** Superseded facts get
   `is_current = false`. Duplicate entities get `merged_into` set.
3. **Confidence is always in [0, 1]** (or NULL). Jev `Score`/`Noul` outputs are
   normalized to this range when stored.
4. **The graph views are the contract.** NetworkX code must read
   `v_graph_nodes`/`v_graph_edges`, never raw tables, so merged entities are
   automatically excluded.
5. **Idempotent ingestion.** Re-ingesting the same file (same sha256) must not
   duplicate documents or re-extract.

## 5. Code conventions

- Type hints on all public functions; stdlib `dataclasses` over pydantic for now.
- Heavy imports (`torch`, `spacy`, `gliner2`, `pandas`, `psycopg`) go inside
  functions, not module top-level, so `--help` and tests work without extras.
- Every extractor backend is optional at runtime: fail with an actionable
  install hint (`pip install -e .[extraction]`), never a traceback.
- Errors are logged with `logging`, not printed.
- Tests: `tests/` must run without Docker, Postgres, or network. Use the
  baseline extractor and pure-python fixtures.

## 6. Workflow protocol

1. Read `PROGRESS.xml`; pick a task with `status="todo"` (or your assigned one).
2. Set `status="in-progress"` with your identifier before working.
3. Do the work in small slices. Keep each slice consistent with
   `docs/ARCHITECTURE.md`.
4. **Commit each finished slice before starting the next one.** A slice is one
   logical change: a module, a migration, a decision record, or a test that
   passes. Do not batch a whole task into a single commit at the end.
5. Before each commit: run the relevant tests (`py -m pytest` when the slice
   touches extraction or the gate). Do not commit a tree you know is broken.
6. Commit message: what changed and why, one or two sentences. Never commit
   `.env`, tokens, or credentials (see §8).
7. **Push that commit to `origin` before starting the next slice.** Do not wait
   to be asked. Use the Basic `x-access-token` header in §8 (`GITHUB_PAT` from
   `.env`). A Bearer header is rejected. Do not store the token in
   `.git/config`, put it in the remote URL, or print it. If the push fails,
   stop and say so. Do not force-push unless the user asks.
8. When the task is done: update task `status="done"`, add a one-line `notes`,
   bump the root `updated` attribute, and commit and push that bookkeeping with
   the slice it describes (or as its own commit if the code is already committed).
9. If you made a significant technical choice, add a decision record to
   `docs/DECISIONS.md` and reference its ID in the task notes. Commit the
   record with the code it justifies.
10. New research findings go into `docs/RESEARCH.md` (with source URLs), not
    into code comments.

## 7. Known open questions

- Jev API key / account: **resolved 2026-10-11 (P5.6)**. The client contract is
  live-verified through the OpenRouter mirror (`model` → `typesafe/jev-1.13-20260917`)
  for all three answer kinds; see `docs/RESEARCH.md` §4b. Provide one of
  `PRATT_LLM_API_KEY` (Pratt router, `/v1/systemone`, model `pratt-jev`),
  `SYSTEMONE_API_KEY` (native TypeSafe), or `OPENROUTER_API_KEY` (OpenRouter
  mirror) in `.env` — the endpoint and model are inferred (D12); without a key
  the pipeline runs offline on exact/alias/trigram resolution and
  extractor-confidence promotion. `PRATT_LLM_API_KEY` also drives LLM escalation
  (`/v1/chat/completions`, model `pratt-medium`).
- `codesoda/systemone` (`s1 serve`) can host GLiNER2 behind a Jev-compatible
  API for fully-local development; evaluate it before scaling hosted calls.

## 8. Credentials & secrets

Secrets live in `.env` at the repo root — **gitignored, never committed** — and
are loaded into the environment by `entity_graph.config.load_config()`
(`python-dotenv`). Known variables:

| Variable in `.env` | Used for |
| --- | --- |
| `GITHUB_PAT` | GitHub personal access token for pushing this repo |
| `PRATT_LLM_API_KEY` | Pratt model router: LLM escalation (`pratt-medium`) + Jev (`pratt-jev`) |
| `OPENROUTER_API_KEY` | Fallback provider for the LLM and Jev (OpenRouter mirror) |

The Git remote is `origin` → `https://github.com/tedngai/entity-graph.git`.
Every finished slice is pushed there (§6). The token is `GITHUB_PAT` in `.env`
(classic `ghp_` PAT). Do **not** commit it, print it, or embed it in
`.git/config`.

This agent shell cannot use the Git Credential Manager dialog. A Bearer
`http.extraheader` is rejected (`invalid credentials`). Push with Basic auth,
username `x-access-token`, password the PAT, and scrub the token from any
captured output before showing it:

```powershell
$pat = $null
foreach ($line in Get-Content .env) {
  if ($line -match '^\s*GITHUB_PAT\s*=\s*(.+)\s*$') {
    $pat = $Matches[1].Trim().Trim('"').Trim("'")
  }
}
$pair = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("x-access-token:${pat}"))
$env:GIT_TERMINAL_PROMPT = '0'
git -c "http.extraheader=AUTHORIZATION: basic $pair" push origin HEAD
git status -sb   # confirm the branch is not ahead of origin
```

PowerShell may wrap git's stderr as a `NativeCommandError` even when the push
succeeds. Trust `git status -sb`, not that wrapper.
