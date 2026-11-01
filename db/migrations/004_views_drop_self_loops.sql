-- Keep self-loops out of the graph contract. A curation merge can turn an edge
-- between two duplicates into winner -> winner (the S-1's signature block gave
-- `Anthropic SUBSIDIARY_OF Anthropic` once `PBC` was merged into `Anthropic`).
-- The relationship row and its facts stay (invariant #2); only the views skip
-- it, and an entity whose only edge is a self-loop counts as isolated.
-- Idempotent; applied by `entity-graph init-db` after db/schema.sql. See D23.

CREATE OR REPLACE VIEW v_graph_nodes AS
SELECT e.id,
       e.canonical_name AS name,
       e.entity_type    AS type,
       e.description,
       e.attributes
FROM entities e
WHERE e.merged_into IS NULL
  AND NOT e.is_hidden
  AND EXISTS (SELECT 1 FROM relationships r
              WHERE (r.source_id = e.id OR r.target_id = e.id)
                AND r.source_id <> r.target_id);

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
JOIN entities se ON se.id = r.source_id AND se.merged_into IS NULL AND NOT se.is_hidden
JOIN entities te ON te.id = r.target_id AND te.merged_into IS NULL AND NOT te.is_hidden
WHERE r.source_id <> r.target_id;
