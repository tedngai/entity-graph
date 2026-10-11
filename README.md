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
| Extraction (primary) | GLiNER2 | Zero-shot NER + classification + relation extraction in one forward pass; runs locally |
| Extraction (fallback/baseline) | spaCy / heuristic baseline | Tokenization, sentence splits; the baseline lets the pipeline run with zero heavy deps |
| Typed decisions | Jev (TypeSafe System One) | Calibrated `Choice`/`Score`/`Noul` decisions for entity alignment, type arbitration, relation validation |
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
rejects relation proposals whose evidence Jev refutes (`--no-validate` skips
it), and `entity-graph curator` lists uncertain pairs awaiting review. Without a
key the pipeline degrades to exact/alias/trigram resolution.

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
│   ├── extract/               # baseline / spacy / gliner2 extractors + Jev client
│   ├── graph_store.py         # write facts; load graph into NetworkX
│   ├── analyze.py  visualize.py  pipeline.py  cli.py
├── data/samples/              # sample unstructured document
└── tests/
```

## Status

See `PROGRESS.xml`. Research, scaffolding, ingestion, chunking, and the GLiNER2
and Jev paths are implemented and verified end to end (GLiNER2 on CUDA; Jev
against the live API). The spaCy backend and the LLM escalation path (P4.3,
P4.5) remain skeletons marked with `TODO`s.
