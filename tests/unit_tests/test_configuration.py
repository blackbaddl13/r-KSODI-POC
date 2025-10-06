# SPDX-License-Identifier: MIT
import os

import pytest

from react_agent.context import Context


@pytest.fixture(autouse=True)
def _clear_env():
    # Save before each test and restore afterwards
    old = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_context_init() -> None:
    # No ENV: the passed value is retained
    ctx = Context(model="openai/gpt-4o-mini")
    assert ctx.model == "openai/gpt-4o-mini"


def test_context_init_with_env_vars() -> None:
    # ENV set: value is taken from environment
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    ctx = Context()
    assert ctx.model == "openai/gpt-4o-mini"


def test_context_init_env_overrides_param() -> None:
    # In your Context, ENV overrides the parameter
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    ctx = Context(model="openai/gpt-5o-mini")
    assert ctx.model == "openai/gpt-4o-mini"
