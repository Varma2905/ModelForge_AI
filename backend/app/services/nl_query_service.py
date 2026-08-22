import json
import logging
import re
from typing import Any, Dict, List, Optional

from app.agents.dataset_agent import get_llm

logger = logging.getLogger("regression_studio.nl_query_service")

try:
    from langchain_core.messages import HumanMessage, SystemMessage
    LANGCHAIN_SCHEMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_SCHEMA_AVAILABLE = False

MAX_SAMPLE_ROWS_FOR_EXPLANATION = 20

_QUERY_SYSTEM_PROMPT = """You are a database query generator. You are given a real
database schema and a natural-language question. Respond with ONLY a single JSON
object — no markdown fences, no explanation, no extra text before or after it.

Rules:
- Only reference tables/collections and columns/fields that actually appear in the
  schema below. Never invent a name that isn't there.
- The query must be strictly read-only. NEVER generate INSERT, UPDATE, DELETE, DROP,
  ALTER, CREATE, TRUNCATE, or any other statement that modifies data or schema.
- If the question implies an order or ranking (e.g. "top", "highest", "lowest",
  "most", "least", "latest", "oldest", "best", "worst"), the query MUST actually sort
  by the relevant field in the right direction — never leave results unsorted and rely
  on a human (or a later summary) to figure out the order from unsorted data.
- If the engine is "mysql" or "postgresql", respond with exactly:
  {"sql": "SELECT ..."}
  A single SELECT statement, no trailing semicolon, no comments. Use ORDER BY (and
  LIMIT, if the question asks for a specific number of results) to satisfy the rule
  above.
- If the engine is "mongodb", respond with exactly:
  {"collection": "<one of the real collection names>", "filter": {<MongoDB filter
  object, or {} for none>}, "sort": [["field", 1 or -1]], "fields": ["field1", "field2"]}
  Use -1 for descending (highest/most/latest) and 1 for ascending (lowest/least/oldest).
  "fields" may be omitted if all fields are relevant; "sort" should be populated
  whenever the rule above applies.
"""


class NLQueryError(Exception):
    """Mirrors ConnectorError/HFClientError/HFExecutionError's category pattern."""

    def __init__(self, category: str, message: str):
        # category: "not_configured" | "generation_failed"
        self.category = category
        super().__init__(message)


def _extract_json_object(text: str) -> Dict[str, Any]:
    """LLMs reliably wrap JSON in ```json fences despite instructions not to —
    strip those before parsing, and fall back to the first {...} span found."""
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else stripped
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    brace_match = re.search(r"\{.*\}", candidate, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass
    raise NLQueryError("generation_failed", "The AI produced an invalid query. Try rephrasing your question.")


async def generate_query(question: str, engine: str, schema_context: str) -> Dict[str, Any]:
    llm = get_llm()
    if llm is None or not LANGCHAIN_SCHEMA_AVAILABLE:
        raise NLQueryError(
            "not_configured", "AI service is not configured. Please configure the AI API key on the backend."
        )

    human_content = f"Database engine: {engine}\n\nSchema:\n{schema_context}\n\nQuestion: {question}"

    try:
        response = await llm.ainvoke(
            [SystemMessage(content=_QUERY_SYSTEM_PROMPT), HumanMessage(content=human_content)]
        )
    except Exception as e:
        logger.warning(f"NL query generation LLM call failed: {e}")
        raise NLQueryError("generation_failed", "The AI service failed to generate a query. Please try again.")

    parsed = _extract_json_object(response.content)

    if engine == "mongodb":
        if "collection" not in parsed:
            raise NLQueryError("generation_failed", "The AI response did not specify a collection.")
    else:
        if "sql" not in parsed or not isinstance(parsed["sql"], str):
            raise NLQueryError("generation_failed", "The AI response did not contain a SQL query.")

    return parsed


async def explain_result(
    question: str, columns: List[str], rows: List[List[Any]], row_count: int
) -> Optional[str]:
    llm = get_llm()
    if llm is None or not LANGCHAIN_SCHEMA_AVAILABLE:
        return None

    sample = rows[:MAX_SAMPLE_ROWS_FOR_EXPLANATION]
    lines = [
        f"Question: {question}",
        f"Columns: {', '.join(columns)}",
        f"Rows returned: {row_count}",
        f"Sample of actual returned rows (up to {MAX_SAMPLE_ROWS_FOR_EXPLANATION}):",
    ]
    for row in sample:
        lines.append(str(dict(zip(columns, row))))

    try:
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content="You explain real database query results in plain, concise English for a "
                    "business user. Only describe what is actually in the data provided — never invent "
                    "numbers, trends, or rows that aren't shown to you."
                ),
                HumanMessage(content="\n".join(lines)),
            ]
        )
        return response.content
    except Exception as e:
        logger.warning(f"NL result explanation LLM call failed: {e}")
        return None
