-- ============================================================================
-- entity-graph :: PostgreSQL schema
-- ----------------------------------------------------------------------------
-- Design principles:
--   1. Facts are REIFIED: every claim (subject, relation, object) is stored with
--      evidence text, provenance (document), extractor, confidence, and an
--      optional temporal validity window. The `relationships` table is the
--      aggregated graph view; `facts` is the evidence trail behind it.
--   2. Entities are canonical. Surface mentions live in `mentions` with char
--      offsets so every extraction decision can be audited back to the text.
--   3. Nothing is hard-deleted. Superseded facts get is_current = false; merged
--      (duplicate) entities keep a merged_into pointer.
--   4. `v_graph_nodes` / `v_graph_edges` views are the contract consumed by
--      NetworkX (see src/entity_graph/graph_store.py).
-- ============================================================================

-- Fuzzy name matching for entity resolution (built-in contrib extension).
CREATE EXTENSION IF NOT EXISTS pg_trgm;
-- Optional: enable when you add name/context embeddings for entity resolution.
-- Requires the pgvector image (see docker-compose.yml):
-- CREATE EXTENSION IF NOT EXISTS vector;

-- ----------------------------------------------------------------------------
-- documents: source material ingested from unstructured files
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    doc_key       TEXT UNIQUE NOT NULL,              -- stable key: path or URI
    title         TEXT,
    source_uri    TEXT,
    content_type  TEXT,                              -- txt | md | pdf | html ...
    raw_text      TEXT,
    checksum      TEXT NOT NULL UNIQUE,              -- sha256; ingest dedupe (ON CONFLICT)
    metadata      JSONB NOT NULL DEFAULT '{}',
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ----------------------------------------------------------------------------
-- entities: canonical real-world referents
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type    TEXT,                             -- PERSON | ORG | PLACE | CONCEPT | EVENT | ...
    description    TEXT,
    attributes     JSONB NOT NULL DEFAULT '{}',
    -- embedding     vector(1536),                  -- uncomment with pgvector
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    merged_into    BIGINT REFERENCES entities (id)   -- soft merge pointer (dedupe)
);
CREATE INDEX IF NOT EXISTS idx_entities_name_trgm
    ON entities USING gin (canonical_name gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities (entity_type);

CREATE TABLE IF NOT EXISTS entity_aliases (
    entity_id  BIGINT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    alias      TEXT NOT NULL,
    source     TEXT,                                  -- extractor | curator | rule
    PRIMARY KEY (entity_id, alias)
);
CREATE INDEX IF NOT EXISTS idx_aliases_trgm ON entity_aliases USING gin (alias gin_trgm_ops);

-- ----------------------------------------------------------------------------
-- mentions: where a surface form occurred (span-level provenance)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mentions (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id   BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    entity_id     BIGINT REFERENCES entities (id),   -- NULL until resolved
    surface_form  TEXT NOT NULL,
    char_start    INT NOT NULL,
    char_end      INT NOT NULL,
    entity_type   TEXT,                              -- type proposed by extractor
    extractor     TEXT NOT NULL,                     -- baseline | spacy | gliner2 | llm
    confidence    REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_mentions_document ON mentions (document_id);
CREATE INDEX IF NOT EXISTS idx_mentions_entity   ON mentions (entity_id);

-- ----------------------------------------------------------------------------
-- relationships: aggregated graph edges (what NetworkX loads)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relationships (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id   BIGINT NOT NULL REFERENCES entities (id),
    target_id   BIGINT NOT NULL REFERENCES entities (id),
    relation    TEXT NOT NULL,                       -- WORKS_FOR | LOCATED_IN | RELATED_TO | ...
    is_directed BOOLEAN NOT NULL DEFAULT TRUE,
    attributes  JSONB NOT NULL DEFAULT '{}',
    UNIQUE (source_id, target_id, relation)
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships (source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships (target_id);

-- ----------------------------------------------------------------------------
-- relation_proposals: staged candidate edges from extraction. `build` promotes
-- them into relationships + facts once mentions are resolved to entities.
-- `validation` holds the Jev Noul score (does the evidence assert the claim?);
-- `status` is pending -> promoted | rejected (rejected = Jev refuted it).
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relation_proposals (
    id             BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id    BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    source_surface TEXT NOT NULL,
    target_surface TEXT NOT NULL,
    relation       TEXT NOT NULL,
    evidence_text  TEXT,
    extractor      TEXT NOT NULL,
    confidence     REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    validation     REAL CHECK (validation IS NULL OR validation BETWEEN 0 AND 1),
    status         TEXT NOT NULL DEFAULT 'pending'
                   CHECK (status IN ('pending', 'promoted', 'rejected')),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_proposals_document ON relation_proposals (document_id);
CREATE INDEX IF NOT EXISTS idx_proposals_pending
    ON relation_proposals (status) WHERE status = 'pending';

-- ----------------------------------------------------------------------------
-- curator_queue: uncertain entity pairs routed by Jev alignment (Score == 1).
-- A human (or a stricter pass) resolves these; nothing is merged automatically.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS curator_queue (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    entity_id    BIGINT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    candidate_id BIGINT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
    score        REAL,
    sidecars     JSONB NOT NULL DEFAULT '{}',
    reason       TEXT,
    status       TEXT NOT NULL DEFAULT 'pending'
                 CHECK (status IN ('pending', 'merged', 'rejected')),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (entity_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_curator_pending
    ON curator_queue (status) WHERE status = 'pending';

-- ----------------------------------------------------------------------------
-- facts: reified claims with evidence + validity (the "key facts" store)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS facts (
    id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    relationship_id BIGINT NOT NULL REFERENCES relationships (id) ON DELETE CASCADE,
    document_id     BIGINT NOT NULL REFERENCES documents (id),
    statement       TEXT NOT NULL,                   -- "Maria Chen works for GreenGrid"
    evidence_text   TEXT,                            -- supporting quote from the document
    extractor       TEXT NOT NULL,
    confidence      REAL CHECK (confidence IS NULL OR confidence BETWEEN 0 AND 1),
    valid_from      DATE,
    valid_to        DATE,
    is_current      BOOLEAN NOT NULL DEFAULT TRUE,   -- flip when superseded, never delete
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_facts_relationship ON facts (relationship_id);
CREATE INDEX IF NOT EXISTS idx_facts_document     ON facts (document_id);
CREATE INDEX IF NOT EXISTS idx_facts_current      ON facts (is_current) WHERE is_current;

-- ----------------------------------------------------------------------------
-- Graph views: the stable contract consumed by NetworkX.
-- Merged entities (merged_into IS NOT NULL) are excluded.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_graph_nodes AS
SELECT e.id,
       e.canonical_name AS name,
       e.entity_type    AS type,
       e.description,
       e.attributes
FROM entities e
WHERE e.merged_into IS NULL;

CREATE OR REPLACE VIEW v_graph_edges AS
SELECT r.id              AS relationship_id,
       r.source_id,
       r.target_id,
       r.relation,
       r.is_directed,
       r.attributes,
       se.canonical_name AS source_name,
       se.entity_type    AS source_type,
       te.canonical_name AS target_name,
       te.entity_type    AS target_type,
       (SELECT count(*) FROM facts f
         WHERE f.relationship_id = r.id AND f.is_current) AS fact_count
FROM relationships r
JOIN entities se ON se.id = r.source_id AND se.merged_into IS NULL
JOIN entities te ON te.id = r.target_id AND te.merged_into IS NULL;
