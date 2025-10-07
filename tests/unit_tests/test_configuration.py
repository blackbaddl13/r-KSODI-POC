# SPDX-License-Identifier: MIT
import os
from collections.abc import Iterator
from typing import Callable, TypeVar, cast

import pytest

from react_agent.context import Context

F = TypeVar("F", bound=Callable[..., object])
FIXTURE = cast(Callable[[F], F], pytest.fixture(autouse=True))

@FIXTURE
def _clear_env() -> Iterator[None]:
    # Save before each test and restore afterwards
    old: dict[str, str] = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def test_context_init() -> None:
    ctx = Context(model="openai/gpt-4o-mini")
    assert ctx.model == "openai/gpt-4o-mini"


def test_context_init_with_env_vars() -> None:
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    ctx = Context()
    assert ctx.model == "openai/gpt-4o-mini"


def test_context_init_env_overrides_param() -> None:
    os.environ["MODEL"] = "openai/gpt-4o-mini"
    ctx = Context(model="openai/gpt-5o-mini")
    assert ctx.model == "openai/gpt-4o-mini"
