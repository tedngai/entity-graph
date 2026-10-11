"""LLM extractor backend — the escalation path (PROGRESS P4.5).

Per D2 an LLM is used **only to escalate** documents/spans the cheap pass flags
as uncertain, not as the primary extractor. This module provides the raw LLM
backend (OpenAI-compatible chat completions; OpenRouter by default) and a
``EscalatingExtractor`` wrapper lives in ``escalation.py``.

Configuration (see ``.env.example``)::

    PRATT_LLM_API_KEY=...      # preferred: Pratt router (see below)
    LLM_MODEL=pratt-medium     # default when PRATT_LLM_API_KEY is set
    LLM_URL=https://llm.pratt.edu/v1/chat/completions
    # Fallbacks (used only when no PRATT_LLM_API_KEY):
    #   LLM_API_KEY=...          # falls back to OPENROUTER_API_KEY
    #   LLM_MODEL=openai/gpt-4o-mini
    #   LLM_URL=https://openrouter.ai/api/v1/chat/completions

The model is asked for a strict JSON document; the reply is parsed defensively
(code fences tolerated) and entity surfaces are anchored back to char offsets in
the source text. Anything that cannot be anchored is dropped so provenance
(invariant #1) always holds.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from ..chunking import split_sentences
from ..config import load_config
from .base import ENTITY_TYPES, ExtractionResult, Mention, RelationProposal

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise information-extraction engine. "
    "Return ONLY a JSON object, with no prose and no code fences."
)

USER_PROMPT = """\
Extract the entities and the relationships between them from the text.

Return JSON with exactly this shape:
{{
  "entities": [{{"text": "<verbatim surface>", "type": "PERSON|ORG|PLACE|CONCEPT|EVENT|DATE|MISC"}}],
  "relations": [{{"source": "<entity text>", "target": "<entity text>",
                  "relation": "UPPER_SNAKE_CASE", "evidence": "<verbatim sentence>"}}]
}}

Rules:
- "text" must appear verbatim in the text below.
- Use short, general relation names (FOUNDED, LEADS, WORKS_FOR, LOCATED_IN,
  PARTNERED_WITH, ACQUIRED, INVESTED_IN, MEMBER_OF, SUBSIDIARY_OF, RELATED_TO).
- Only include a relation if the text supports it.

TEXT:
{text}
"""

MENTION_CONFIDENCE = 0.6
RELATION_CONFIDENCE = 0.6


def parse_json_object(content: str) -> dict:
    """Parse a JSON object from a model reply, tolerating code fences."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text[:4].lower() == "json":
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in LLM reply: {content[:200]!r}")
    return json.loads(text[start : end + 1])


def _find_span(text: str, surface: str) -> tuple[int, int] | None:
    """First case-sensitive (then insensitive) occurrence of ``surface``."""
    if not surface:
        return None
    index = text.find(surface)
    if index == -1:
        index = text.lower().find(surface.lower())
    if index == -1:
        return None
    return index, index + len(surface)


def _sentence_at(text: str, position: int) -> str:
    for start, end in split_sentences(text):
        if start <= position < end:
            return text[start:end].strip()
    return ""


class LlmExtractor:
    name = "llm"

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 url: str | None = None, timeout: float = 60.0) -> None:
        config = load_config()
        self.model = model if model is not None else config.llm_model
        self.api_key = api_key if api_key is not None else config.llm_api_key
        self.url = url if url is not None else config.llm_url
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if config.openrouter_referer:
            self.headers["HTTP-Referer"] = config.openrouter_referer
        if config.openrouter_title:
            self.headers["X-OpenRouter-Title"] = config.openrouter_title

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        if not self.enabled:
            raise RuntimeError(
                "No LLM API key. Set LLM_API_KEY or OPENROUTER_API_KEY."
            )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={**self.headers, "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:800]
            except Exception:  # noqa: BLE001 - best-effort diagnostics
                pass
            raise RuntimeError(
                f"LLM call failed: HTTP {exc.code} {exc.reason}. {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"LLM call failed: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response: {body!r}") from exc

    def extract(self, text: str, doc_key: str = "") -> ExtractionResult:
        prompt = USER_PROMPT.format(text=text)
        content = self._chat(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": prompt}]
        )
        data = parse_json_object(content)

        mentions: list[Mention] = []
        surfaces: set[str] = set()
        for item in data.get("entities", []) or []:
            if not isinstance(item, dict):
                continue
            surface = str(item.get("text", "")).strip()
            span = _find_span(text, surface)
            if span is None:
                continue
            entity_type = str(item.get("type", "MISC")).upper()
            if entity_type not in ENTITY_TYPES:
                entity_type = "MISC"
            mentions.append(
                Mention(surface=surface, entity_type=entity_type,
                        start=span[0], end=span[1], confidence=MENTION_CONFIDENCE)
            )
            surfaces.add(surface)

        relations: list[RelationProposal] = []
        for item in data.get("relations", []) or []:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation = str(item.get("relation", "")).strip().upper().replace(" ", "_")
            if not (source and target and relation) or source == target:
                continue
            evidence = str(item.get("evidence", "")).strip()
            if not evidence:
                span = _find_span(text, source)
                evidence = _sentence_at(text, span[0]) if span else ""
            relations.append(
                RelationProposal(source_surface=source, target_surface=target,
                                 relation=relation, evidence=evidence,
                                 confidence=RELATION_CONFIDENCE)
            )

        log.info("LLM (%s) extracted %d mentions, %d relations for %s",
                 self.model, len(mentions), len(relations), doc_key or "document")
        return ExtractionResult(doc_key=doc_key, mentions=mentions, relations=relations)
