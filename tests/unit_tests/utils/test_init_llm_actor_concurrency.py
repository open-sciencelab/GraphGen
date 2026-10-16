"""Ray synthesizer actor concurrency remains bounded and opt-in."""
from __future__ import annotations

import sys

import pytest

from graphgen.common.init_llm import LLMFactory


@pytest.mark.parametrize(
    ("configured", "expected"),
    [(None, None), ("1", None), ("4", 4)],
)
def test_actor_max_concurrency_is_opt_in(
    monkeypatch: pytest.MonkeyPatch, configured: str | None, expected: int | None
) -> None:
    options_seen: dict = {}

    class FakeRemoteMethod:
        def remote(self):
            return "ready"

    class FakeHandle:
        ready = FakeRemoteMethod()

    class FakeRemote:
        def options(self, **options):
            options_seen.update(options)
            return self

        def remote(self, backend, config):
            return FakeHandle()

    class FakeRay:
        @staticmethod
        def get_actor(_name):
            raise ValueError("missing")

        @staticmethod
        def remote(_actor):
            return FakeRemote()

        @staticmethod
        def get(value):
            return value

    monkeypatch.setitem(sys.modules, "ray", FakeRay)
    if configured is None:
        monkeypatch.delenv("GRAPHGEN_LLM_ACTOR_MAX_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("GRAPHGEN_LLM_ACTOR_MAX_CONCURRENCY", configured)

    result = LLMFactory.create_llm("synthesizer", "fake", {"num_gpus": 0})

    assert result is not None
    assert options_seen["get_if_exists"] is True
    assert options_seen.get("max_concurrency") == expected
