import logging
from contextvars import Context

from graphgen.utils.log import logger


def test_context_aware_logger_handles_ray_context_without_operator_context(caplog):
    """LLM Ray actors may log without inheriting an operator's ContextVar."""
    with caplog.at_level(logging.WARNING):
        Context().run(logger.warning, "empty LLM response from actor")

    assert "empty LLM response from actor" in caplog.text
