"""Integração opt-in com a API real do Gemini Robotics ER (#276).

Só roda com ``GEMINI_API_KEY`` exportada no ambiente; o ``.env`` não é lido aqui de
propósito, para que a suíte normal nunca envie imagens para fora da máquina.
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("google.genai")

from fixtures import payload_with_blobs  # noqa: E402
from visual_perception.application.scene_context import validate_scene_response  # noqa: E402
from visual_perception.config import MultimodalReasoningConfig  # noqa: E402
from visual_perception.infrastructure.adapters.gemini_reasoning_backend import (  # noqa: E402
    DEFAULT_GEMINI_ROBOTICS_ER_MODEL,
    GEMINI_API_KEY_ENV,
    GeminiRoboticsReasoningAdapter,
)

pytestmark = pytest.mark.skipif(
    not os.environ.get(GEMINI_API_KEY_ENV), reason=f"{GEMINI_API_KEY_ENV} não exportada; integração remota é opt-in"
)


# Confere o caminho real ponta a ponta: a resposta de cena passa pela mesma
# validação de schema da produção e a telemetria registra tokens.
def test_live_scene_response_satisfies_the_scene_contract() -> None:
    """Consulta o Gemini com uma imagem sintética e valida o contract de cena."""
    adapter = GeminiRoboticsReasoningAdapter()
    config = MultimodalReasoningConfig(backend="gemini_robotics_er", checkpoint=DEFAULT_GEMINI_ROBOTICS_ER_MODEL)

    response = adapter.analyze_scene(payload_with_blobs(width=64, height=48), config)

    validate_scene_response(response)
    (call,) = adapter.calls
    assert call.succeeded and call.prompt_tokens
