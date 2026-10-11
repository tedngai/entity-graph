-- Hide isolated entities (no incident relationship) from the graph contract.
-- Idempotent (CREATE OR REPLACE VIEW); applied by `entity-graph init-db` after
-- db/schema.sql. Entities are never deleted (invariant #2), only filtered from
-- the view, so analysis and visualization see the real network. See D18.

CREATE OR REPLACE VIEW v_graph_nodes AS
SELECT e.id,
       e.canonical_name AS name,
       e.entity_type    AS type,
       e.description,
       e.attributes
FROM entities e
WHERE e.merged_into IS NULL
  AND EXISTS (SELECT 1 FROM relationships r
              WHERE r.source_id = e.id OR r.target_id = e.id);
