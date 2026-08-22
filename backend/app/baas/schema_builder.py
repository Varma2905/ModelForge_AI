import json
import logging
import re
from typing import Any, Dict, List

from app.agents.dataset_agent import get_llm

logger = logging.getLogger("regression_studio.baas.schema_builder")

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_SCHEMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_SCHEMA_AVAILABLE = False

ALLOWED_COLUMN_TYPES = {"string", "number", "boolean", "date"}

_SCHEMA_SYSTEM_PROMPT = """You design simple table schemas for a generic backend-as-a-
service data store. Given a plain-English description, respond with ONLY a JSON array
of column objects — no markdown fences, no explanation, no extra text.

Each column object must be exactly: {"name": "<snake_case identifier>", "type": "<one
of: string, number, boolean, date>"}. Do not include an "id" column — one is added
automatically. Keep it to the columns the description actually implies; do not invent
unrelated columns."""


class SchemaProposalError(Exception):
    def __init__(self, category: str, message: str):
        # category: "not_configured" | "generation_failed"
        self.category = category
        super().__init__(message)


def _extract_json_array(text: str) -> Any:
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    bracket_match = re.search(r"\[.*\]", candidate, re.DOTALL)
    if bracket_match:
        try:
            return json.loads(bracket_match.group(0))
        except json.JSONDecodeError:
            pass
    raise SchemaProposalError("generation_failed", "The AI produced an invalid schema. Try rephrasing, or define columns manually.")


async def propose_schema(description: str) -> List[Dict[str, str]]:
    """Ephemeral — never persists. Mirrors the propose-then-confirm split
    data_sources.py already uses for test-vs-save connections."""
    llm = get_llm()
    if llm is None or not LANGCHAIN_SCHEMA_AVAILABLE:
        raise SchemaProposalError(
            "not_configured", "AI service is not configured. Define your table's columns manually instead."
        )

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_SCHEMA_SYSTEM_PROMPT), HumanMessage(content=description)]
        )
    except Exception as e:
        logger.warning(f"BaaS schema proposal LLM call failed: {e}")
        raise SchemaProposalError("generation_failed", "The AI service failed to propose a schema. Please try again.")

    parsed = _extract_json_array(response.content)
    if not isinstance(parsed, list) or not parsed:
        raise SchemaProposalError("generation_failed", "The AI did not propose any columns. Try rephrasing.")

    columns: List[Dict[str, str]] = []
    seen_names = set()
    for col in parsed:
        if not isinstance(col, dict):
            continue
        name = str(col.get("name", "")).strip().lower()
        col_type = str(col.get("type", "")).strip().lower()
        if not name or name in seen_names or name == "id":
            continue
        if col_type not in ALLOWED_COLUMN_TYPES:
            col_type = "string"
        seen_names.add(name)
        columns.append({"name": name, "type": col_type})

    if not columns:
        raise SchemaProposalError("generation_failed", "The AI did not propose any usable columns. Try rephrasing.")

    return columns
