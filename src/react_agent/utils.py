# SPDX-License-Identifier: MIT
"""Utility & helper functions."""

import re
from typing import Any, Type, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# Use native OpenAI wrapper to ensure usage + streaming.
ChatOpenAI: Type[Any] | None
try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI
    ChatOpenAI = _ChatOpenAI
except Exception:
    ChatOpenAI = None


def get_message_text(msg: BaseMessage) -> str:
    """Extract text from a BaseMessage, handling various content formats."""
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("text", "") or ""
    parts: list[str] = []
    for c in (content or []):
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, dict):
            parts.append(c.get("text", "") or "")
        else:
            parts.append(str(c))
    return "".join(parts).strip()


def load_chat_model(fully_specified_name: str, **kwargs: Any) -> BaseChatModel:
    """Load a chat model by name, with optional streaming support."""
    provider, model = _split_provider_model(fully_specified_name)
    streaming = bool(kwargs.get("streaming", False))

    if provider == "openai" and ChatOpenAI is not None:
        return cast(BaseChatModel, ChatOpenAI(model=model, stream_usage=True, streaming=streaming))

    return cast(
        BaseChatModel,
        init_chat_model(model, model_provider=provider, **{k: v for k, v in kwargs.items() if k != "streaming"}),
    )


def _split_provider_model(name: str) -> tuple[str, str]:
    if "/" in name:
        p, m = name.split("/", maxsplit=1)
        return p.strip(), m.strip()
    return "openai", name.strip()


# ---------- Prompt header stripping (for CC BY headers etc.) ----------

_CUT_MARKERS: tuple[str, ...] = (
    '---8<--- CUT HERE (build strips above) ---8<---',
    '---8<---',
    '<!-- CUT HERE -->',
)
_YAML_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_prompt_header(text: str) -> str:
    """Strip YAML frontmatter or custom cut markers from a prompt string."""
    if not text:
        return text
    if text.startswith("---\n"):
        fm = _YAML_FRONTMATTER_RE.match(text)
        if fm:
            return text[fm.end():].lstrip()
    first_hit_idx: int | None = None
    first_hit_len = 0
    for marker in _CUT_MARKERS:
        i = text.find(marker)
        if i >= 0 and (first_hit_idx is None or i < first_hit_idx):
            first_hit_idx = i
            first_hit_len = len(marker)
    if first_hit_idx is not None:
        return text[first_hit_idx + first_hit_len:].lstrip()
    return text


def _strip_message_content(
    content: str | list[Any] | dict[str, Any]
) -> str | list[str | dict[str, Any]]:
    """Normalize + strip headers across common LangChain content forms.

    Guarantees: returns either `str` or `list[str | dict[str, Any]]` (no bare dict).
    """
    if isinstance(content, str):
        return strip_prompt_header(content)

    if isinstance(content, dict):
        # normalize single dict → list[dict]
        d: dict[str, Any] = dict(content)
        if "text" in d and isinstance(d["text"], str):
            d["text"] = strip_prompt_header(d["text"])
        return [d]

    if isinstance(content, list):
        out: list[str | dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                out.append(strip_prompt_header(part))
            elif isinstance(part, dict):
                p: dict[str, Any] = dict(part)
                if "text" in p and isinstance(p["text"], str):
                    p["text"] = strip_prompt_header(p["text"])
                out.append(p)
            else:
                # normalize unknown parts to string to satisfy typing & robustness
                out.append(str(part))
        return out

    # fallback (shouldn't happen): coerce to str
    return str(content)


def strip_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Strip YAML frontmatter or custom cut markers from a list of messages."""
    out: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, SystemMessage | HumanMessage | AIMessage):
            new_content = _strip_message_content(m.content)
            out.append(type(m)(
                content=new_content,
                additional_kwargs=getattr(m, "additional_kwargs", {}),
                response_metadata=getattr(m, "response_metadata", {}),
                name=getattr(m, "name", None),
            ))
        else:
            out.append(m)
    return out
