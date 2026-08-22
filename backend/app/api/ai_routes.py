import json
import logging
import re
import time
from collections import defaultdict, deque

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal, Optional

from app.auth.dependencies import get_current_user
from app.database.mongodb import db_client
from app.agents.dataset_agent import DatasetAnalysisAgent, get_llm
from app.agents.ml_agent import MLModelExplanationAgent
from app.agents.explanation_agent import StatisticalInterpretationAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.report_agent import ReportAgent
from app.utils.response import ok

logger = logging.getLogger("regression_studio.ai_routes")

try:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    LANGCHAIN_SCHEMA_AVAILABLE = True
except ImportError:
    LANGCHAIN_SCHEMA_AVAILABLE = False

router = APIRouter(prefix="", tags=["Agentic AI Systems"])

# ── Chat Models ───────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    source: str  # "llm" | "fallback"


class AIExplainRequest(BaseModel):
    model_id: str

class AIExplainResponse(BaseModel):
    summary: str
    insights: List[str]
    full_report: str

async def run_ai_explanation_pipeline(model_id: str, user_id: str) -> Dict[str, Any]:
    # 1. Fetch model document, scoped to the requesting user
    model_doc = await db_client.find_one("models", {"_id": model_id})
    if not model_doc or model_doc.get("user_id") != user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with ID {model_id} not found."
        )

    # 2. Fetch dataset document
    dataset_doc = await db_client.find_one("datasets", {"_id": model_doc["dataset_id"]})
    if not dataset_doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset with ID {model_doc['dataset_id']} referencing model was not found."
        )

    # 3. Instantiate agents
    dataset_agent = DatasetAnalysisAgent()
    ml_agent = MLModelExplanationAgent()
    explanation_agent = StatisticalInterpretationAgent()
    rec_agent = RecommendationAgent()
    report_agent = ReportAgent()

    # 4. Generate summaries asynchronously
    df_summary = {
        "name": dataset_doc["name"],
        "rows": dataset_doc["row_count"],
        "columns": dataset_doc["col_count"],
        "column_names": dataset_doc["columns"],
        "data_types": dataset_doc["data_types"]
    }

    # Build components
    dataset_analysis = await dataset_agent.analyze(df_summary)
    model_explanation = await ml_agent.explain(model_doc["model"], model_doc.get("hyperparameters"))
    stats_explanation = await explanation_agent.explain(
        metrics=model_doc["metrics"],
        stats=model_doc["statistical_analysis"],
        features=model_doc["features"],
        target=model_doc["target"]
    )
    recommendations = await rec_agent.recommend(
        model_name=model_doc["model"],
        metrics=model_doc["metrics"],
        stats=model_doc["statistical_analysis"],
        features=model_doc["features"],
        target=model_doc["target"]
    )

    # Combine into full report
    full_report = await report_agent.generate_full_report(
        dataset_name=dataset_doc["name"],
        model_name=model_doc["model"],
        metrics=model_doc["metrics"],
        dataset_analysis=dataset_analysis,
        model_explanation=model_explanation,
        stats_explanation=stats_explanation,
        recommendations=recommendations
    )

    # 5. Extract summary and insights for JSON response
    r2 = model_doc["metrics"].get("R2", 0.0)
    summary_text = f"The model trained with {model_doc['model']} achieved an R² score of {r2:.2f}."

    # Try parsing the OLS interpretations for significant feature insights
    insights = []
    bullet_points = re.findall(r'^\s*[-•]\s+(.*?)$', stats_explanation + "\n" + recommendations, re.MULTILINE)
    for bp in bullet_points:
        clean_bp = bp.replace("**", "").replace("`", "").strip()
        if len(clean_bp) > 10 and clean_bp not in insights:
            insights.append(clean_bp)

    # Fallback default insights if parsing did not find bullet points
    if not insights:
        insights = [
            f"The features {', '.join(model_doc['features'])} were used to fit the regression line.",
            f"The Mean Absolute Error (MAE) was calculated to be {model_doc['metrics'].get('MAE', 0.0):,.2f}.",
            "Review diagnostic recommendations to check feature p-values and try alternative algorithms."
        ]

    # Slice to top 4 insights
    insights = insights[:4]

    # Save the AI report markdown to the model document for later retrieval (e.g. for PDF)
    await db_client.update_one(
        "models",
        {"_id": model_id},
        {"$set": {"ai_explanation": full_report}}
    )

    # Save report metadata to reports collection
    report_doc = {
        "_id": f"rep_{model_id}",
        "user_id": user_id,
        "model_id": model_id,
        "summary": summary_text,
        "insights": insights,
        "full_report": full_report,
        "created_at": pd.Timestamp.now().isoformat()
    }
    await db_client.insert_one("reports", report_doc)

    return {
        "summary": summary_text,
        "insights": insights,
        "full_report": full_report
    }

@router.post("/ai/explain")
async def ai_explain(request: AIExplainRequest, current_user: dict = Depends(get_current_user)):
    result = await run_ai_explanation_pipeline(request.model_id, current_user["_id"])
    return ok(result)

@router.get("/ai-explanation")
async def ai_explain_latest(current_user: dict = Depends(get_current_user)):
    # Find current user's latest model
    models = await db_client.find_many("models", {"user_id": current_user["_id"]})
    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No models found. Train a model first to get AI explanations."
        )

    sorted_models = sorted(models, key=lambda x: x.get("created_at", ""), reverse=True)
    latest_model_id = sorted_models[0]["_id"]

    result = await run_ai_explanation_pipeline(latest_model_id, current_user["_id"])
    return ok(result)


# ── Chatbot Endpoint ──────────────────────────────────────────────────────────
CHAT_SYSTEM_PROMPT = """You are an expert AI assistant specializing in regression analysis,
machine learning, and data science. You help users understand statistical concepts,
choose regression models, interpret results (R², RMSE, MAE, p-values, coefficients),
and improve their models.

Be concise, clear, and practical. Use bullet points when listing options.
If a question is unrelated to data science or regression, politely redirect the
conversation back to your area of expertise.
"""

def _chat_fallback(message: str) -> str:
    """Rule-based fallback when no LLM is configured."""
    l = message.lower()
    if ("model" in l and "choose" in l) or "which model" in l or "best model" in l:
        return (
            "**Model Selection Guide:**\n"
            "- **Linear/Ridge/Lasso** → When you expect a linear relationship and need interpretability.\n"
            "- **Polynomial Regression** → When the data shows a curved trend.\n"
            "- **Random Forest** → Excellent all-rounder for non-linear, complex datasets.\n"
            "- **SVR** → Small-to-medium datasets with non-linear boundaries.\n"
            "- **Elastic Net** → Many correlated features; combines Ridge + Lasso benefits.\n\n"
            "Start simple (Linear), then compare RMSE/R² on a test set before escalating complexity."
        )
    if "r2" in l or "r²" in l or "r-squared" in l or "r squared" in l:
        return (
            "**R² (R-Squared)** measures how much variance in the target your model explains:\n"
            "- **0.90 – 1.00** → Excellent fit ✅\n"
            "- **0.70 – 0.90** → Strong fit, reliable for forecasting ✅\n"
            "- **0.50 – 0.70** → Moderate fit; consider adding more features ⚠️\n"
            "- **< 0.50** → Weak fit; model needs significant improvement ❌\n\n"
            "Always validate R² on **unseen test data** — a high training R² can indicate overfitting."
        )
    if "overfit" in l:
        return (
            "**Overfitting** happens when your model memorizes training data instead of learning patterns.\n"
            "**Signs:** Very high training R², much lower test R².\n"
            "**Fixes:**\n"
            "- Add more training data\n"
            "- Use regularization (Ridge / Lasso / ElasticNet)\n"
            "- Reduce model complexity (lower polynomial degree / tree depth)\n"
            "- Apply cross-validation during training"
        )
    if "ridge" in l and "lasso" in l:
        return (
            "**Ridge vs Lasso:**\n"
            "- **Ridge (L2):** Shrinks all coefficients smoothly toward zero. Keeps all features. Best when many features contribute.\n"
            "- **Lasso (L1):** Can zero out coefficients completely → automatic feature selection. Best when only a few features matter.\n"
            "- **Elastic Net:** Combines both — gets grouping effect of Ridge + sparsity of Lasso."
        )
    if "p-value" in l or "p value" in l or "significance" in l or "significant" in l:
        return (
            "**P-values** tell you whether a feature statistically influences the target:\n"
            "- **p < 0.05** → Feature is **statistically significant** (95% confidence) ✅\n"
            "- **p ≥ 0.05** → Feature may not genuinely affect the target ⚠️ (consider dropping it)\n\n"
            "A low p-value doesn't mean a feature is practically important — also look at the coefficient size."
        )
    if "rmse" in l or "mae" in l or "error" in l:
        return (
            "**Error Metrics:**\n"
            "- **RMSE (Root Mean Squared Error):** Penalizes large errors heavily. Good for catching outlier predictions.\n"
            "- **MAE (Mean Absolute Error):** Average absolute difference. More robust to outliers.\n"
            "- **Lower is always better.** Compare both metrics across models to find the best fit."
        )
    if "feature" in l and ("engineer" in l or "selection" in l or "import" in l):
        return (
            "**Feature Engineering Tips:**\n"
            "- Drop features with p-value ≥ 0.05 (statistically insignificant)\n"
            "- Create interaction terms (e.g., `Area × Bedrooms`) if joint effects exist\n"
            "- Apply log-transform to skewed features\n"
            "- Use StandardScaler before training Ridge, Lasso, SVR, or ElasticNet\n"
            "- Check correlation matrix — drop one of any pair with correlation > 0.85"
        )
    return (
        "Great question! I'm your regression analysis assistant. I can help you with:\n"
        "- **Model selection** (Linear, Ridge, Lasso, Random Forest, SVR…)\n"
        "- **Interpreting results** (R², RMSE, MAE, p-values, coefficients)\n"
        "- **Fixing overfitting / underfitting**\n"
        "- **Feature engineering & preprocessing tips**\n\n"
        "Try asking: *'Which model should I choose?'* or *'How do I interpret my R² score?'*"
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage, current_user: dict = Depends(get_current_user)):
    """LLM-powered chatbot endpoint for regression analysis Q&A."""
    llm = get_llm(provider=request.provider, model=request.model)

    if llm and LANGCHAIN_SCHEMA_AVAILABLE:
        try:
            system_msg = SystemMessage(content=CHAT_SYSTEM_PROMPT)
            human_msg = HumanMessage(content=request.message)
            response = await llm.ainvoke([system_msg, human_msg])
            return ChatResponse(reply=response.content, source="llm")
        except Exception as e:
            logger.warning(f"Chat LLM call failed: {e}. Using fallback.")

    # Fallback: rule-based responses
    return ChatResponse(reply=_chat_fallback(request.message), source="fallback")


# ── Real-time Streaming Chat (ChatGPT-style) ────────────────────────────────
MAX_MESSAGE_CHARS = 6000
MAX_HISTORY_MESSAGES = 40
MAX_CONTEXT_TURNS = 16  # most recent turns actually sent to the LLM

# Simple in-memory sliding-window rate limiter, per authenticated user.
# Fine for the single-process dev/deploy this app already assumes elsewhere
# (see db_client's JSON fallback) — not shared across processes.
_RATE_LIMIT_WINDOW_SEC = 60
_RATE_LIMIT_MAX_REQUESTS = 20
_rate_limit_state: Dict[str, deque] = defaultdict(deque)


def _check_rate_limit(user_id: str) -> None:
    now = time.time()
    q = _rate_limit_state[user_id]
    while q and now - q[0] > _RATE_LIMIT_WINDOW_SEC:
        q.popleft()
    if len(q) >= _RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="You're sending messages too quickly. Please wait a moment and try again.",
        )
    q.append(now)


class AIChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AIChatRequest(BaseModel):
    messages: List[AIChatTurn] = Field(..., min_length=1)
    model_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


async def _build_regression_context_text(model_id: str, user_id: str) -> Optional[str]:
    """Builds a compact, human-readable summary of one of the user's trained
    models for grounding chat answers. Returns None (silently) if the model
    doesn't exist or isn't owned by this user — context is a nice-to-have,
    not a reason to fail the whole chat request."""
    model_doc = await db_client.find_one("models", {"_id": model_id})
    if not model_doc or model_doc.get("user_id") != user_id:
        return None

    dataset_doc = await db_client.find_one("datasets", {"_id": model_doc.get("dataset_id")})
    metrics = model_doc.get("metrics", {}) or {}
    features = model_doc.get("features", []) or []
    target = model_doc.get("target", "")
    stats = model_doc.get("statistical_analysis", {}) or {}
    p_values = stats.get("p_values", {}) or {}
    importance = (model_doc.get("chart_data", {}) or {}).get("feature_importance", []) or []
    top_importance = sorted(importance, key=lambda x: abs(x.get("value", 0)), reverse=True)[:5]

    lines = [
        f"Dataset: {model_doc.get('dataset_name', 'unknown')}"
        + (f" ({dataset_doc['row_count']} rows)" if dataset_doc and dataset_doc.get("row_count") else ""),
        f"Model type: {model_doc.get('model', 'unknown')}",
        f"Target column: {target}",
        f"Features ({len(features)}): {', '.join(features) if features else 'none'}",
    ]

    test_size = (model_doc.get("split") or {}).get("test_size")
    if test_size:
        lines.append(f"Train/test split: {(1 - test_size) * 100:.0f}% train / {test_size * 100:.0f}% test")

    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if numeric_metrics:
        lines.append("Metrics on held-out test set: " + ", ".join(f"{k}={v:.4f}" for k, v in numeric_metrics.items()))

    if top_importance:
        lines.append(
            "Top features by coefficient magnitude: "
            + ", ".join(f"{i['feature']} ({i['value']:.3f})" for i in top_importance if "feature" in i)
        )

    if p_values and features:
        significant = [f for f in features if p_values.get(f, 1) <= 0.05]
        insignificant = [f for f in features if p_values.get(f, 1) > 0.05]
        if significant:
            lines.append(f"Statistically significant features (p<=0.05): {', '.join(significant)}")
        if insignificant:
            lines.append(f"Not statistically significant (p>0.05): {', '.join(insignificant)}")

    return "\n".join(lines)


def _classify_llm_error(exc: Exception) -> tuple[str, str]:
    """Maps a raw provider exception to a safe (error_type, user_message) pair.
    Never surfaces the raw exception text — it can contain request internals."""
    msg = str(exc).lower()
    if any(tok in msg for tok in ("401", "unauthorized", "invalid api key", "invalid_api_key", "authenticate", "permission_denied", "403")):
        return "auth", "Unable to authenticate with the AI service. Please check the backend API configuration."
    if any(tok in msg for tok in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")):
        return "rate_limit", "The AI service is receiving too many requests right now. Please wait a moment and try again."
    if any(tok in msg for tok in ("timeout", "timed out", "connection", "network")):
        return "network", "Connection failed. Please try again."
    return "server", "The AI service encountered an error. Please try again."


@router.post("/ai/chat")
async def ai_chat_stream(payload: AIChatRequest, current_user: dict = Depends(get_current_user)):
    """Real-time, streaming, multi-turn chat endpoint. Streams Server-Sent
    Events of the form `data: {"delta": "..."}` as the LLM generates tokens,
    followed by a final `data: {"done": true}`, or `data: {"error": "...",
    "error_type": "..."}` if the provider call fails mid-stream."""
    user_id = current_user["_id"]
    _check_rate_limit(user_id)

    if len(payload.messages) > MAX_HISTORY_MESSAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Conversation history is too long (max {MAX_HISTORY_MESSAGES} messages).",
        )
    if payload.messages[-1].role != "user":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The last message must be from the user.")
    for turn in payload.messages:
        content = turn.content.strip()
        if not content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message content must not be empty.")
        if len(content) > MAX_MESSAGE_CHARS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Message is too long (max {MAX_MESSAGE_CHARS} characters).",
            )

    if not LANGCHAIN_SCHEMA_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured. Please configure the AI API key on the backend.",
        )

    llm = get_llm(provider=payload.provider, model=payload.model)
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured. Please configure the AI API key on the backend.",
        )

    system_prompt = CHAT_SYSTEM_PROMPT
    if payload.model_id:
        try:
            context_text = await _build_regression_context_text(payload.model_id, user_id)
        except Exception as e:
            logger.warning(f"Failed to build regression context for chat: {e}")
            context_text = None
        if context_text:
            system_prompt += (
                "\n\nThe user is currently looking at this trained regression model. "
                "Use it to answer questions like 'why is my R2 low' concretely:\n" + context_text
            )

    lc_messages: List[Any] = [SystemMessage(content=system_prompt)]
    for turn in payload.messages[-MAX_CONTEXT_TURNS:]:
        if turn.role == "user":
            lc_messages.append(HumanMessage(content=turn.content))
        else:
            lc_messages.append(AIMessage(content=turn.content))

    async def event_stream():
        try:
            async for chunk in llm.astream(lc_messages):
                delta = getattr(chunk, "content", "") or ""
                if delta:
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.warning(f"AI chat stream failed: {e}")
            error_type, error_text = _classify_llm_error(e)
            yield f"data: {json.dumps({'error': error_text, 'error_type': error_type})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
