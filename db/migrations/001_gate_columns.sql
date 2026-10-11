-- Upgrade databases created before the JointIE / gate columns existed.
-- Idempotent. Applied by `entity-graph init-db` after db/schema.sql.

ALTER TABLE relation_proposals ADD COLUMN IF NOT EXISTS source_type TEXT;
ALTER TABLE relation_proposals ADD COLUMN IF NOT EXISTS target_type TEXT;
ALTER TABLE relation_proposals ADD COLUMN IF NOT EXISTS modality TEXT;
ALTER TABLE relation_proposals ADD COLUMN IF NOT EXISTS gate TEXT;
