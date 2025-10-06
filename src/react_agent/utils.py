# SPDX-License-Identifier: MIT
"""Utility & helper functions."""

from typing import Tuple, Any, List, Union
import re

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage

# Use native OpenAI wrapper to ensure usage + streaming.
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


def load_chat_model(fully_specified_name: str, **kwargs: Any) -> BaseChatModel:
    """Load 'provider/model'. For OpenAI: stream_usage=True; streaming passthrough."""
    provider, model = _split_provider_model(fully_specified_name)
    streaming = bool(kwargs.get("streaming", False))

    if provider == "openai" and ChatOpenAI is not None:
        return ChatOpenAI(model=model, stream_usage=True, streaming=streaming)

    # Other providers via init_chat_model; forward extra kwargs if supported.
    return init_chat_model(model, model_provider=provider, **{k: v for k, v in kwargs.items() if k != "streaming"})


def _split_provider_model(name: str) -> Tuple[str, str]:
    """Split 'provider/model'; default to OpenAI if omitted."""
    if "/" in name:
        p, m = name.split("/", maxsplit=1)
        return p.strip(), m.strip()
    return "openai", name.strip()


# ---------- Prompt header stripping (for CC BY headers etc.) ----------

# Supported cut markers (first match wins).
_CUT_MARKERS: tuple[str, ...] = (
    '---8<--- CUT HERE (build strips above) ---8<---',
    '---8<---',
    '<!-- CUT HERE -->',
)

# YAML front matter: remove leading block if present.
_YAML_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def strip_prompt_header(text: str) -> str:
    """
    Remove license/meta header above a cut marker or YAML front matter.
    Idempotent: safe to call multiple times.
    """
    if not text:
        return text

    # 1) YAML front matter
    if text.startswith("---\n"):
        m = _YAML_FRONTMATTER_RE.match(text)
        if m:
            return text[m.end():].lstrip()

    # 2) CUT markers
    first_hit_idx = None
    first_hit_len = 0
    for m in _CUT_MARKERS:
        i = text.find(m)
        if i >= 0 and (first_hit_idx is None or i < first_hit_idx):
            first_hit_idx = i
            first_hit_len = len(m)

    if first_hit_idx is not None:
        return text[first_hit_idx + first_hit_len :].lstrip()

    return text


def _strip_message_content(content: Union[str, list, dict]) -> Union[str, list, dict]:
    """Apply strip_prompt_header across common LangChain content forms."""
    if isinstance(content, str):
        return strip_prompt_header(content)

    if isinstance(content, dict):
        if "text" in content and isinstance(content["text"], str):
            d = dict(content)
            d["text"] = strip_prompt_header(d["text"])
            return d
        return content

    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(strip_prompt_header(part))
            elif isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
                p = dict(part)
                p["text"] = strip_prompt_header(p["text"])
                out.append(p)
            else:
                out.append(part)
        return out

    return content


def strip_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Return new messages with stripped headers for System/Human/AI.
    ToolMessage and others are left untouched.
    """
    out: List[BaseMessage] = []
    for m in messages:
        if isinstance(m, (SystemMessage, HumanMessage, AIMessage)):
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
