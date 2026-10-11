"""Configuration loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # no-op when .env is absent


@dataclass(frozen=True)
class Config:
    # PostgreSQL (matches docker-compose.yml)
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_db: str = "entity_graph"
    pg_user: str = "postgres"
    pg_password: str = "postgres"

    # Extraction backend: baseline | spacy | gliner2
    extraction_backend: str = "baseline"

    # GLiNER2 (primary extractor). device: auto | cpu | cuda
    gliner_model: str = "fastino/gliner2.5-base-v1"
    gliner_device: str = "auto"
    gliner_threshold: float = 0.4

    # Local gate when Jev is not configured (GLiNER2.5-Decide). Device falls
    # back to GLINER_DEVICE. The model is loaded only at build time, and only
    # when no System One key is set.
    decide_model: str = "fastino/GLiNER2.5-Decide"
    decide_device: str = "auto"

    # LLM escalation (PROGRESS P4.5; opt-in). OpenAI-compatible chat completions.
    # Endpoint/model/key are inferred in load_config(): a bare PRATT_LLM_API_KEY
    # selects the Pratt router (POST /v1/chat/completions, model pratt-medium);
    # otherwise OpenRouter (model openai/gpt-4o-mini) is used.
    llm_url: str = "https://openrouter.ai/api/v1/chat/completions"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-4o-mini"
    escalate: bool = False
    escalation_confidence: float = 0.5

    # Long-document chunking: sentence windows with overlap (see chunking.py)
    chunk_max_sentences: int = 8
    chunk_overlap_sentences: int = 2
    chunk_max_chars: int = 1200

    # Jev / System One (TypeSafe) typed decisions.
    # Endpoint/model/key are inferred in load_config() in this order:
    #   SYSTEMONE_API_KEY  -> native TypeSafe (POST /v1/systemone, jev-latest)
    #   PRATT_LLM_API_KEY  -> Pratt router   (POST /v1/systemone, pratt-jev)
    #   OPENROUTER_API_KEY -> OpenRouter mirror (/api/alpha/decisions,
    #                         model typesafe/jev-1.13)
    systemone_url: str = "https://api.typesafe.ai/v1/systemone"
    systemone_api_key: str = ""
    systemone_model: str = "jev-latest"
    # Optional OpenRouter attribution headers (sent only when non-empty).
    openrouter_referer: str = ""
    openrouter_title: str = "entity-graph"

    # Paths (relative to the repo root; run CLI commands from there)
    schema_path: Path = Path("db/schema.sql")
    out_dir: Path = Path("out")

    @property
    def dsn(self) -> str:
        return (
            f"host={self.pg_host} port={self.pg_port} dbname={self.pg_db} "
            f"user={self.pg_user} password={self.pg_password}"
        )


def load_config() -> Config:
    env = os.environ.get
    pratt_key = env("PRATT_LLM_API_KEY", "")
    openrouter_key = env("OPENROUTER_API_KEY", "")
    systemone_key = env("SYSTEMONE_API_KEY", "")

    # LLM escalation (P4.5). Precedence: explicit LLM_* overrides > Pratt
    # router (when PRATT_LLM_API_KEY is set) > OpenRouter (historical default).
    if pratt_key:
        llm_url_default = "https://llm.pratt.edu/v1/chat/completions"
        llm_model_default = "pratt-medium"
        llm_key_default = pratt_key
    else:
        llm_url_default = "https://openrouter.ai/api/v1/chat/completions"
        llm_model_default = "openai/gpt-4o-mini"
        llm_key_default = openrouter_key

    # Jev credentials/endpoint. Precedence: explicit SYSTEMONE_* overrides >
    # native TypeSafe key > Pratt router > OpenRouter mirror.
    if systemone_key:
        jev_url_default = "https://api.typesafe.ai/v1/systemone"
        jev_model_default = "jev-latest"
        jev_key_default = systemone_key
    elif pratt_key:
        jev_url_default = "https://llm.pratt.edu/v1/systemone"
        jev_model_default = "pratt-jev"
        jev_key_default = pratt_key
    else:
        jev_url_default = "https://openrouter.ai/api/alpha/decisions"
        jev_model_default = "typesafe/jev-1.13"
        jev_key_default = openrouter_key

    return Config(
        pg_host=env("PG_HOST", "localhost"),
        pg_port=int(env("PG_PORT", "5432")),
        pg_db=env("PG_DB", "entity_graph"),
        pg_user=env("PG_USER", "postgres"),
        pg_password=env("PG_PASSWORD", "postgres"),
        extraction_backend=env("EXTRACTION_BACKEND", "baseline"),
        gliner_model=env("GLINER_MODEL", "fastino/gliner2.5-base-v1"),
        gliner_device=env("GLINER_DEVICE", "auto"),
        gliner_threshold=float(env("GLINER_THRESHOLD", "0.4")),
        decide_model=env("DECIDE_MODEL", "fastino/GLiNER2.5-Decide"),
        decide_device=env("DECIDE_DEVICE", env("GLINER_DEVICE", "auto")),
        llm_url=env("LLM_URL", llm_url_default),
        llm_api_key=env("LLM_API_KEY", "") or llm_key_default,
        llm_model=env("LLM_MODEL", llm_model_default),
        escalate=env("EXTRACTION_ESCALATE", "").strip().lower() in ("1", "true", "yes"),
        escalation_confidence=float(env("ESCALATION_CONFIDENCE", "0.5")),
        chunk_max_sentences=int(env("CHUNK_MAX_SENTENCES", "8")),
        chunk_overlap_sentences=int(env("CHUNK_OVERLAP_SENTENCES", "2")),
        chunk_max_chars=int(env("CHUNK_MAX_CHARS", "1200")),
        systemone_url=env("SYSTEMONE_URL", jev_url_default),
        systemone_api_key=env("SYSTEMONE_API_KEY", "") or jev_key_default,
        systemone_model=env("SYSTEMONE_MODEL", jev_model_default),
        openrouter_referer=env("OPENROUTER_REFERER", ""),
        openrouter_title=env("OPENROUTER_TITLE", "entity-graph"),
        schema_path=Path(env("SCHEMA_PATH", "db/schema.sql")),
        out_dir=Path(env("OUT_DIR", "out")),
    )
