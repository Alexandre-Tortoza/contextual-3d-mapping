"""Adapter de composição com fallback permanente de raciocínio multimodal (#292)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendExecutionError
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import EvidenceSlot
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionRelationRequest

#: Teto de views usado pelo Qwen local quando ele assume como fallback, e
#: `fallback_region_views` não foi setado explicitamente na config. Um crop
#: focado no sujeito (foreground obrigatório) mais a visão geral da cena —
#: metade das até 4 views da referência de qualidade.
_QWEN_FALLBACK_REGION_VIEWS = (
    EvidenceSlot.MASKED_SUBJECT.value,
    EvidenceSlot.SCENE_CONDITIONED.value,
)


# Envolve dois reasoners do mesmo port para sobreviver ao esgotamento do
# primário sem abortar o run inteiro. Ao contrário de
# FallbackDenseFeatureExtractionAdapter (que tenta o primário em toda
# chamada, barato porque local), aqui a troca é permanente pelo resto da
# vida do adapter: o primário típico é uma API remota de cota limitada
# (Gemini), cujo erro mais comum depois de #292 é cota esgotada — que não se
# resolve tentando de novo, e insistir custaria os retries+backoff do
# próprio backend primário (`config.max_retries`) em toda chamada seguinte
# ao longo de um run com milhares delas.
class FallbackMultimodalReasoningAdapter:
    """Satisfaz o mesmo port dos demais reasoners, com fallback permanente sob falha."""

    # Recebe os dois reasoners já prontos; a escolha de qual classe concreta
    # cada um é vem da factory, não daqui — este adapter só sabe compor.
    def __init__(self, primary: Any, fallback: Any) -> None:
        """Inicializa o decorator com os reasoners primário e alternativo."""
        self._primary = primary
        self._fallback = fallback
        self._active = primary
        #: Exposto para diagnóstico/telemetria: se o run já trocou de backend.
        self.switched = False

    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do reasoner ativo para o contexto da cena."""
        return self._call(lambda active, cfg: active.analyze_scene(image, cfg), config)

    def discover_concepts(
        self, image: ImagePayload, config: MultimodalReasoningConfig, *, max_concepts: int
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do reasoner ativo com conceitos a localizar."""
        return self._call(
            lambda active, cfg: active.discover_concepts(image, cfg, max_concepts=max_concepts), config
        )

    def analyze_region(
        self, request: RegionReasoningRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do reasoner ativo para uma região da imagem."""
        return self._call(lambda active, cfg: active.analyze_region(request, cfg), config)

    def analyze_relation(
        self, request: RegionRelationRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna a resposta JSON bruta do reasoner ativo sobre a relação entre duas regiões."""
        return self._call(lambda active, cfg: active.analyze_relation(request, cfg), config)

    # Executa a chamada contra o reasoner ativo, trocando para o fallback (e
    # refazendo a MESMA chamada, para não perder a consulta que originou a
    # troca) na primeira falha do primário. Uma falha do fallback já ativo
    # propaga normalmente: não há mais para onde cair.
    def _call(
        self, invoke: Callable[[Any, MultimodalReasoningConfig], dict[str, Any]], config: MultimodalReasoningConfig,
    ) -> dict[str, Any]:
        """Invoca `invoke` contra o reasoner ativo, com troca permanente sob falha do primário."""
        try:
            return invoke(self._active, self._effective_config(config))
        except BackendExecutionError:
            if self._active is not self._primary:
                raise
            self._active = self._fallback
            self.switched = True
            return invoke(self._active, self._effective_config(config))

    # Deriva a config do backend de fallback a partir da config declarada
    # pelo primário, no mesmo espírito de FallbackDenseFeatureExtractionAdapter:
    # backend/checkpoint trocados, e region_views reduzido por default quando
    # o fallback é o Qwen local (ver `_QWEN_FALLBACK_REGION_VIEWS`).
    def _effective_config(self, config: MultimodalReasoningConfig) -> MultimodalReasoningConfig:
        """Retorna a config a usar na chamada, dado qual reasoner está ativo."""
        if self._active is self._primary:
            return config
        region_views = config.fallback_region_views
        if region_views is None and config.fallback_backend == "qwen_vl":
            region_views = _QWEN_FALLBACK_REGION_VIEWS
        return dataclasses.replace(
            config,
            backend=config.fallback_backend,
            checkpoint=config.fallback_checkpoint,
            region_views=region_views if region_views is not None else config.region_views,
            fallback_backend=None,
            fallback_checkpoint=None,
            fallback_region_views=None,
        )
