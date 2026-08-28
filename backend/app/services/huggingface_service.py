"""Hugging Face Inference API service — the ONLY LLM provider for the AI
Assistant (both the real-time chat endpoint and the report-explanation
agent pipeline). See HF_TOKEN / HF_MODEL in .env.example.

There is deliberately no fallback to any other LLM provider here: if the
Hugging Face call fails, callers surface a friendly error (see
ai_routes.py's _classify_llm_error) instead of silently trying a different
provider.

`HuggingFaceChatModel` exposes just the two async call shapes the rest of
the app already used with its previous LangChain-based providers
(`ainvoke([...]) -> object with .content` and
`async for chunk in astream([...]): chunk.content`), so none of the
report-explanation agents or the streaming chat endpoint needed to change
when this provider was swapped in — they still build plain
`langchain_core.messages` objects (SystemMessage/HumanMessage/AIMessage)
and read `.content` off the result exactly as before.
"""
import logging
import os
from typing import Any, AsyncIterator, List, Optional

from huggingface_hub import AsyncInferenceClient

logger = logging.getLogger("regression_studio.huggingface_service")

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 30.0

# langchain_core message `.type` -> Hugging Face chat-completion role. Kept
# as a plain dict lookup rather than importing langchain_core here, so this
# module has no LangChain dependency of its own — only the callers do.
_ROLE_BY_MESSAGE_TYPE = {"system": "system", "human": "user", "ai": "assistant"}


class _ChatResult:
    """Minimal stand-in for a LangChain AIMessage/AIMessageChunk. Every
    caller in this codebase only ever reads `.content` off whatever
    ainvoke()/astream() returns, so a full BaseChatModel isn't needed."""

    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


def _to_hf_messages(messages: List[Any]) -> List[dict]:
    """Converts a list of langchain_core message objects into the plain
    role/content dicts the Hugging Face chat-completions API expects."""
    out = []
    for m in messages:
        role = _ROLE_BY_MESSAGE_TYPE.get(getattr(m, "type", None), "user")
        out.append({"role": role, "content": m.content})
    return out


class HuggingFaceChatModel:
    """Thin async chat-completion client for the Hugging Face Inference API."""

    def __init__(self, model: Optional[str] = None, token: Optional[str] = None, timeout: float = DEFAULT_TIMEOUT):
        self.model = model or os.getenv("HF_MODEL") or DEFAULT_MODEL
        self.token = token or os.getenv("HF_TOKEN")
        self._client = AsyncInferenceClient(model=self.model, token=self.token, timeout=timeout)

    async def ainvoke(self, messages: List[Any]) -> _ChatResult:
        hf_messages = _to_hf_messages(messages)
        completion = await self._client.chat_completion(messages=hf_messages, max_tokens=DEFAULT_MAX_TOKENS)
        content = ""
        if completion.choices:
            content = completion.choices[0].message.content or ""
        return _ChatResult(content)

    async def astream(self, messages: List[Any]) -> AsyncIterator[_ChatResult]:
        hf_messages = _to_hf_messages(messages)
        stream = await self._client.chat_completion(
            messages=hf_messages, max_tokens=DEFAULT_MAX_TOKENS, stream=True
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield _ChatResult(delta)


def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> Optional[HuggingFaceChatModel]:
    """Returns a Hugging Face-backed chat model, or None if HF_TOKEN isn't
    configured (or the client fails to construct). `provider` is accepted
    only for call-site backward compatibility — Hugging Face is the sole
    supported provider now, so it's ignored. `model` overrides HF_MODEL for
    this one call, when a caller (e.g. a request body) specifies one."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return None
    try:
        return HuggingFaceChatModel(model=model, token=token)
    except Exception:
        logger.exception("Failed to initialize the Hugging Face chat model")
        return None
