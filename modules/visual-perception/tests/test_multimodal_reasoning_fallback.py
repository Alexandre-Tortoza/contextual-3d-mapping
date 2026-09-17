"""Testa o fallback permanente de raciocínio multimodal (#292)."""

from __future__ import annotations

import pytest

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendExecutionError
from visual_perception.infrastructure.adapters.multimodal_reasoning_fallback import (
    FallbackMultimodalReasoningAdapter,
)


# Reasoner primário que sempre falha, como uma API remota com a cota esgotada.
class _AlwaysFailingReasoner:
    """Simula um backend primário permanentemente indisponível."""

    def __init__(self) -> None:
        """Conta quantas vezes cada método foi chamado."""
        self.calls = 0

    def analyze_scene(self, image, config):  # noqa: ANN001
        """Levanta sempre, como uma cota remota já esgotada."""
        self.calls += 1
        raise BackendExecutionError("cota esgotada")

    def discover_concepts(self, image, config, *, max_concepts):  # noqa: ANN001
        """Levanta sempre, como uma cota remota já esgotada."""
        self.calls += 1
        raise BackendExecutionError("cota esgotada")

    def analyze_region(self, request, config):  # noqa: ANN001
        """Levanta sempre, como uma cota remota já esgotada."""
        self.calls += 1
        raise BackendExecutionError("cota esgotada")

    def analyze_relation(self, request, config):  # noqa: ANN001
        """Levanta sempre, como uma cota remota já esgotada."""
        self.calls += 1
        raise BackendExecutionError("cota esgotada")


# Reasoner alternativo que sempre responde e registra a config recebida.
class _WorkingReasoner:
    """Simula o backend local assumindo depois da troca."""

    def __init__(self) -> None:
        """Conta chamadas e guarda a última config recebida por método."""
        self.calls = 0
        self.last_config: MultimodalReasoningConfig | None = None

    def analyze_scene(self, image, config):  # noqa: ANN001
        """Responde sempre, registrando a config efetivamente usada."""
        self.calls += 1
        self.last_config = config
        return {"ok": True}

    def discover_concepts(self, image, config, *, max_concepts):  # noqa: ANN001
        """Responde sempre, registrando a config efetivamente usada."""
        self.calls += 1
        self.last_config = config
        return {"max_concepts": max_concepts}

    def analyze_region(self, request, config):  # noqa: ANN001
        """Responde sempre, registrando a config efetivamente usada."""
        self.calls += 1
        self.last_config = config
        return {"request": request}

    def analyze_relation(self, request, config):  # noqa: ANN001
        """Responde sempre, registrando a config efetivamente usada."""
        self.calls += 1
        self.last_config = config
        return {"request": request}


def _config(**overrides) -> MultimodalReasoningConfig:
    """Constrói uma config de referência para o backend primário Gemini."""
    defaults = dict(
        backend="gemini_robotics_er",
        checkpoint="gemini-robotics-er-2-preview",
        fallback_backend="qwen_vl",
        fallback_checkpoint="Qwen/Qwen2.5-VL-3B-Instruct",
    )
    defaults.update(overrides)
    return MultimodalReasoningConfig(**defaults)


# Confirma que a própria chamada que estourou a falha não se perde: ela é
# refeita contra o fallback, em vez de exigir uma segunda chamada do
# consumidor para ter sucesso.
def test_first_call_switches_and_completes_against_fallback() -> None:
    primary = _AlwaysFailingReasoner()
    fallback = _WorkingReasoner()
    adapter = FallbackMultimodalReasoningAdapter(primary, fallback)

    result = adapter.analyze_scene(image=object(), config=_config())

    assert result == {"ok": True}
    assert primary.calls == 1
    assert fallback.calls == 1
    assert adapter.switched is True


# Confirma que, uma vez trocado, o primário nunca é tentado de novo nesta
# instância — insistir contra uma cota esgotada só gastaria os retries do
# próprio backend primário em toda chamada seguinte.
def test_subsequent_calls_never_retry_the_primary() -> None:
    primary = _AlwaysFailingReasoner()
    fallback = _WorkingReasoner()
    adapter = FallbackMultimodalReasoningAdapter(primary, fallback)

    adapter.analyze_scene(image=object(), config=_config())
    adapter.analyze_scene(image=object(), config=_config())
    adapter.discover_concepts(image=object(), config=_config(), max_concepts=5)

    assert primary.calls == 1
    assert fallback.calls == 3


# Garante que uma falha do próprio fallback, já ativo, propaga normalmente:
# não há um terceiro backend para tentar.
def test_fallback_failure_after_switch_propagates() -> None:
    primary = _AlwaysFailingReasoner()
    fallback = _AlwaysFailingReasoner()
    adapter = FallbackMultimodalReasoningAdapter(primary, fallback)

    with pytest.raises(BackendExecutionError):
        adapter.analyze_scene(image=object(), config=_config())


# Confirma a derivação da config de fallback: backend/checkpoint trocados, e
# region_views reduzido por default para o Qwen, mesmo sem
# fallback_region_views explícito na config original.
def test_fallback_call_uses_reduced_region_views_by_default_for_qwen() -> None:
    primary = _AlwaysFailingReasoner()
    fallback = _WorkingReasoner()
    adapter = FallbackMultimodalReasoningAdapter(primary, fallback)

    adapter.analyze_scene(image=object(), config=_config(region_views=(
        "masked_subject", "tight_crop", "contextual_crop", "scene_conditioned",
    )))

    assert fallback.last_config is not None
    assert fallback.last_config.backend == "qwen_vl"
    assert fallback.last_config.checkpoint == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert fallback.last_config.region_views == ("masked_subject", "scene_conditioned")
    assert fallback.last_config.fallback_backend is None


# Confirma que um `fallback_region_views` explícito na config vence o default
# reduzido do Qwen.
def test_explicit_fallback_region_views_overrides_the_qwen_default() -> None:
    primary = _AlwaysFailingReasoner()
    fallback = _WorkingReasoner()
    adapter = FallbackMultimodalReasoningAdapter(primary, fallback)

    adapter.analyze_scene(
        image=object(),
        config=_config(fallback_region_views=("tight_crop",)),
    )

    assert fallback.last_config.region_views == ("tight_crop",)
