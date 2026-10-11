# entity-graph

Study, analyze, and visualize complex relationships between entities.

**The pipeline:** unstructured documents → entity & relation extraction →
calibrated fact decisions (Jev / System One) → PostgreSQL fact store →
NetworkX analysis → interactive visualization.

## Stack

| Layer | Choice | Why |
|---|---|---|
| Storage | PostgreSQL (+ `pg_trgm`, optional `pgvector`) | Reified facts with provenance; edge-table graph model; recursive CTEs for traversal |
| Graph analytics | NetworkX | Centrality, community detection, paths — loaded straight from a SQL view |
| Extraction (primary) | GLiNER2.5 JointIE | Typed entities and relations decoded together under endpoint and graph constraints; runs locally |
| Extraction (fallback/baseline) | spaCy / heuristic baseline | Tokenization, sentence splits; the baseline lets the pipeline run with zero heavy deps |
| Typed decisions | Jev (TypeSafe System One), or GLiNER2.5-Decide offline | Calibrated `Choice`/`Score`/`Noul` for alignment and relation validation. No key: a local Decide modality + agreement gate |
| Visualization | pyvis / matplotlib | Interactive HTML graphs; static figures for reports |

See `docs/RESEARCH.md` for the full research behind these choices and
`docs/DECISIONS.md` for the decision records.

## Quickstart (Windows / PowerShell)

```powershell
# 1. Database
docker compose up -d

# 2. Python environment (Python 3.11+)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

# 3. Create schema
Copy-Item .env.example .env   # then edit if needed
entity-graph init-db

# 4. Run the full pipeline on the bundled sample document
entity-graph run --input data/samples

# 5. Analyze + visualize (writes out/graph.html)
entity-graph analyze
entity-graph visualize
```

`EXTRACTION_BACKEND=baseline` (the default) needs no model downloads. For real
NER, install the extraction extras and select a backend:

```powershell
pip install -e .[extraction]          # spaCy + gliner2[local] (pulls torch)
python -m spacy download en_core_web_sm   # for the spaCy fallback
# Windows: PyPI installs CPU-only torch; for CUDA add the PyTorch index build:
pip install torch --index-url https://download.pytorch.org/whl/cu130
$env:EXTRACTION_BACKEND = "gliner2"   # uses fastino/gliner2.5-base-v1 (GLINER_DEVICE=auto)
```

Backends: `baseline` (zero-dep), `spacy` (fast local fallback), `gliner2`
(primary, GPU-friendly), and `llm` (OpenAI-compatible, escalation only).
`GLINER_DEVICE` (`auto` | `cpu` | `cuda`), `GLINER_MODEL`, and `GLINER_THRESHOLD`
tune the GLiNER2 backend; see `.env.example`.

**LLM escalation.** Add `--escalate` (or `EXTRACTION_ESCALATE=true`) to re-extract
documents the cheap pass flags as uncertain with an LLM, then merge the results:

```powershell
# Pratt router is used when PRATT_LLM_API_KEY is set (model pratt-medium).
# Otherwise OpenRouter (openai/gpt-4o-mini) with LLM_API_KEY / OPENROUTER_API_KEY.
entity-graph run --input data/samples --escalate
```

**Jev (calibrated decisions).** Entity alignment and relation validation run
only when a key is configured — add one to `.env`:

```dotenv
PRATT_LLM_API_KEY=...    # Pratt router (endpoint + model inferred: pratt-jev)
# or OPENROUTER_API_KEY=...  (typesafe/jev-1.13)
# or SYSTEMONE_API_KEY=...   (native TypeSafe; SYSTEMONE_URL/MODEL override)
```

With Jev enabled, `entity-graph resolve` routes fuzzy entity matches through
Jev's alignment verdict (merge / curator / separate), `entity-graph build`
rejects relation proposals whose evidence is not a fact (modality argmax plus
the frozen assertion Noul; `--no-validate` skips it), and `entity-graph curator`
lists uncertain pairs awaiting review. Without a key, resolution stays
exact/alias/trigram and relation proposals are judged by local
GLiNER2.5-Decide when `gliner2` is installed.

**Gold eval (does not tune thresholds).**

```powershell
entity-graph eval
entity-graph eval --predictions predictions.jsonl
```

`predictions.jsonl` rows are `{subject, relation, object}` plus optional
`assertion`, `modality`, `agrees`, `source`. The 0.5 Noul cutoff stays frozen
until a decision record follows a sweep on `data/gold/`.

## Repository layout

```
entity-graph/
├── AGENTS.md                  # operating manual for AI agents working here
├── PROGRESS.xml               # machine-readable progress tracking
├── db/schema.sql              # PostgreSQL schema (single source of truth)
├── docker-compose.yml         # Postgres 16 + pgvector
├── docs/
│   ├── RESEARCH.md            # research findings & sources
│   ├── ARCHITECTURE.md        # data flow, modules, contracts
│   └── DECISIONS.md           # decision records (ADRs)
├── src/entity_graph/
│   ├── config.py  db.py  ingest.py  resolve.py
│   ├── extract/               # baseline / spacy / JointIE / Decide / Jev
│   ├── graph_store.py         # write facts; load graph into NetworkX
│   ├── eval_facts.py          # gold-fact precision/recall (no threshold writes)
│   ├── analyze.py  visualize.py  pipeline.py  cli.py
├── data/samples/              # sample unstructured document
├── data/gold/                 # golden facts for eval (P8.1)
└── tests/
```

## Status

See `PROGRESS.xml`. Research, scaffolding, ingestion, chunking, JointIE
extraction, the Jev and Decide gates, and the gold-fact eval are implemented.
The 0.5 assertion cutoff is frozen (D15) until `entity-graph eval` is run on
more than the bundled sample.
