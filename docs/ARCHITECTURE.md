# ARCHITECTURE.md

## 1. Data flow

```
 unstructured docs (data/)              TypeSafe Jev (System One)
        │                               POST /v1/systemone
        ▼                                        ▲
 ┌──────────────┐   mentions + candidate          │ typed decisions
 │   ingest.py  │   relations (per extractor)     │ Noul 0.95 / Score 2 / Choice
 └──────┬───────┘──────────┐                      │
        │                  ▼                      │
        │        ┌───────────────────┐    ┌───────┴────────┐
        │        │  extract/         │───▶│ extract/jev.py │
        │        │  baseline|spacy|  │    │ align / filter │
        │        │  gliner2          │    └───────┬────────┘
        │        └─────────┬─────────┘            │ verdicts
        │                  │                      ▼
        │                  ▼            ┌──────────────────┐
        │        ┌───────────────────┐   │   resolve.py     │  exact/alias/trigram
        │        │ graph_store.py    │◀──│ (+ merge/curate) │  blocking in SQL
        │        │ upsert entities   │   └──────────────────┘
        │        │ relationships     │
        ▼        │ facts (evidence!) │
 ┌─────────────┐ └─────────┬─────────┘
 │ PostgreSQL  │           │ v_graph_nodes / v_graph_edges
 │ documents   │           ▼
 │ entities    │ ┌──────────────────┐      ┌───────────────┐
 │ mentions    │ │   analyze.py     │      │ visualize.py  │
 │ relationships│ │ NetworkX: degree │      │ pyvis HTML    │
 │ facts       │ │ betweenness,     │      │ out/graph.html│
 └─────────────┘ │ louvain groups   │      └───────────────┘
                 └──────────────────┘
```

Orchestrated by `pipeline.py`, exposed through `cli.py`
(`ingest | preprocess | extract | resolve | build | analyze | visualize | run | init-db`).

## 2. Modules

| Module | Responsibility |
|---|---|
| `config.py` | Loads `.env` → `Config` dataclass (PG, backend, SystemOne settings) |
| `db.py` | psycopg connection + `apply_schema()` (runs `db/schema.sql`) |
| `ingest.py` | Walk input dir → sha256-deduped rows in `documents`; OCR text is cleaned first (D16) |
| `preprocess.py` | OCR cleanup: strip page scaffolding/footnote markers, rejoin pages, de-hyphenate, normalise punctuation |
| `chunking.py` | Sentence-window splitter for long docs; `Chunk` carries absolute char offsets |
| `extract/base.py` | Contract: `Mention`, `RelationProposal`, `ExtractionResult`, `Extractor` protocol + registry |
| `extract/baseline_extractor.py` | Zero-dep heuristic extractor (capitalized spans, same-sentence `RELATED_TO`) — for tests and first runs |
| `extract/spacy_extractor.py` | spaCy backend (skeleton; needs `pip install -e .[extraction]` + model) |
| `extract/ontology.py` | Shared entity/relation schema for JointIE and the Decide gate |
| `extract/gliner_extractor.py` | GLiNER2.5 JointIE backend (primary). Independent `.relations()` only if `joint_ie` is missing |
| `extract/gate.py` | Pure promotion rule (`should_promote`). Does not call a model |
| `extract/decide.py` | Local GLiNER2.5-Decide gate, used only when Jev is not configured |
| `extract/jev.py` | SystemOne client: `decide()` + `align_pair()` + `judge_relations()` (Noul + modality Choice) |
| `eval_facts.py` | Gold-fact scorer. Prints a threshold sweep; never writes the frozen cutoff |
| `resolve.py` | Mention → entity: exact/alias/trigram SQL; creates canonical entities; hook for Jev alignment verdicts |
| `graph_store.py` | Write path (`upsert_*`, `record_fact`) + read path (`load_graph()` → `nx.MultiDiGraph`) |
| `analyze.py` | NetworkX report: size, density, centralities, louvain communities |
| `visualize.py` | pyvis HTML export, node colors by entity type |
| `pipeline.py` / `cli.py` | Orchestration + argparse CLI |

## 3. Contracts

### Extraction contract (`extract/base.py`)
```python
Mention(surface, entity_type, start, end, confidence)     # char offsets into doc text
RelationProposal(source_surface, target_surface, relation, evidence, confidence,
                 source_type="", target_type="")       # ontology labels when known
ExtractionResult(doc_key, mentions, relations)
Extractor.name: str
Extractor.extract(text: str) -> ExtractionResult
```

### Chunking contract (`chunking.py`)
```python
split_sentences(text) -> list[(start, end)]        # spans tile the text
chunk_text(text, *, max_sentences, overlap, max_chars) -> list[Chunk]
Chunk(text, start, end, index)                     # text == source[start:end]
```
`pipeline.extract_document()` extracts long documents window by window, rebases
each mention's offsets to the full document, and collapses overlap duplicates
(mentions by span, relations by `(source, target, relation)`). A single-window
document bypasses chunking entirely. Windows overlap so boundary-straddling
mentions/evidence are still seen; see `docs/DECISIONS.md` D6.

### Database contract
- Writers go through `graph_store.py` only.
- Readers (NetworkX) go through `v_graph_nodes` / `v_graph_edges` only.
- Merged entities never appear in the views (`merged_into IS NULL`).

### Jev decision contract (`extract/jev.py`)
- `decide(state, questions) -> list[dict]` — one POST, many questions.
- `align_pair(a, b) -> "separate" | "curator" | "merge"` — Score routing
  (round to nearest labeled level, **no thresholds**), Noul sidecars per field.
- `validates_relation(evidence, s, rel, o) -> float` — Noul probability.
- `judge_relations(claims) -> {id: RelationJudgment}` — assertion Noul plus a
  modality Choice (`fact` / `possibility` / `denial` / `forecast` / `not_stated`).
- All Jev settings come from config. With no key, entity resolution falls back
  to exact/alias/trigram. Relation promotion then uses the local Decide gate
  (`extract/decide.py`) if `gliner2` is installed, otherwise extractor confidence.
- `should_promote` rejects a scored claim when modality is not `fact`, Decide
  disagrees with the proposed relation, or (Jev only) the assertion Noul is
  below the frozen cutoff. It does not tune that cutoff.

## 4. Confidence semantics
`confidence` columns are always [0,1]. Jev `Noul` outputs map directly; Jev
`Score` levels map to `level / max_level`; extractor confidences pass through.
Facts keep both the extractor and its confidence so quality can be compared
per backend (P8.1 eval set).

## 5. Scaling path (deliberate, not built yet)
1. More docs → chunk long documents into sentence windows (P3.3; implemented in
   `chunking.py` + `pipeline.extract_document`).
2. More edges → materialized aggregates + batch upserts (P6.3).
3. Traversal-heavy SQL → add Apache AGE on the same tables (D1).
4. Bigger graphs → NVIDIA nx-cugraph backend for NetworkX (drop-in).
