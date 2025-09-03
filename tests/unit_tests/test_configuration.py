import os

import pytest

from react_agent.context import Context


@pytest.fixture(autouse=True)
def _clear_env():
    # Vor jedem Test sichern und nachher wiederherstellen
    old = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_context_init() -> None:
    # Kein ENV: der übergebene Wert bleibt erhalten
    ctx = Context(model="openai/gpt-4o-mini")
    assert ctx.model == "openai/gpt-4o-mini"


def test_context_init_with_env_vars() -> None:
    # ENV gesetzt: wird übernommen
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    ctx = Context()
    assert ctx.model == "openai/gpt-4o-mini"


def test_context_init_env_overrides_param() -> None:
    # In deinem Context überschreibt ENV den Parameter
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    ctx = Context(model="openai/gpt-5o-mini")
    assert ctx.model == "openai/gpt-4o-mini"
