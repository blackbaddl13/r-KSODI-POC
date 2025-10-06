"""Utility & helper functions."""

from typing import Tuple
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

# Ensure OpenAI returns usage/token metrics.
try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None  # type: ignore


def get_message_text(msg: BaseMessage) -> str:
    """Extract plain text from a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") or ""
    parts = []
    for c in (content or []):
        parts.append(c if isinstance(c, str) else (c.get("text") or ""))
    return "".join(parts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load 'provider/model'; for OpenAI use stream_usage=True."""
    provider, model = _split_provider_model(fully_specified_name)
    if provider == "openai" and ChatOpenAI is not None:
        return ChatOpenAI(model=model, stream_usage=True)
    return init_chat_model(model, model_provider=provider)


def _split_provider_model(name: str) -> Tuple[str, str]:
    """Split 'provider/model'; default to OpenAI if omitted."""
    if "/" in name:
        p, m = name.split("/", maxsplit=1)
        return p.strip(), m.strip()
    return "openai", name.strip()
