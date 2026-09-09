"""Fake determinístico e livre de GPU para a fronteira de raciocínio multimodal (#164, #165, #189)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendExecutionError
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_reasoning import RegionReasoningRequest, RegionRelationRequest

SceneResponseFn = Callable[[ImagePayload], dict[str, Any]]
RegionResponseFn = Callable[[RegionReasoningRequest], dict[str, Any]]
RelationResponseFn = Callable[[RegionRelationRequest], dict[str, Any]]


# Implementação fake do port MultimodalReasoner. Existe para permitir testar
# e desenvolver o pipeline sem GPU nem VLM real, substituindo o adapter real
# (#189) até que ele esteja pronto.
class FakeMultimodalReasoner:
    """Deriva respostas de cena/região pré-definidas mas sensíveis ao conteúdo.

    Testes que precisam de uma resposta malformada, uma região ambígua
    (múltiplas hipóteses de label), ou uma falha dura de backend injetam
    ``scene_response_fn`` / ``region_response_fn`` em vez de depender da
    heurística de conteúdo.
    """

    # Guarda os hooks opcionais de resposta e o conjunto de regiões que
    # devem simular falha, usados para controlar o comportamento do fake
    # em cenários de teste específicos.
    def __init__(
        self,
        scene_response_fn: SceneResponseFn | None = None,
        region_response_fn: RegionResponseFn | None = None,
        fail_on_region_ids: frozenset[str] = frozenset(),
        relation_response_fn: RelationResponseFn | None = None,
    ) -> None:
        self._scene_response_fn = scene_response_fn
        self._region_response_fn = region_response_fn
        self._fail_on_region_ids = fail_on_region_ids
        self._relation_response_fn = relation_response_fn

    # Gera a análise fake de cena completa: usa o hook injetado se houver,
    # senão deriva um resultado determinístico a partir do brilho médio da
    # imagem.
    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> dict[str, Any]:
        if self._scene_response_fn is not None:
            return self._scene_response_fn(image)
        brightness = float(image.pixels.mean())
        # Emite o mesmo contract ambiental do adapter real (#202): sem prosa
        # livre e sem inventário de objetos, para que a suíte exercite o schema
        # de produção e não um formato que só o fake fala.
        return {
            "scene_type": "well_lit" if brightness > 127 else "dim",
            "environment": "indoor",
            "layout": f"synthetic frame with mean brightness {brightness:.1f}",
            "lighting": "bright" if brightness > 127 else "dim",
            "visibility": "clear",
            "navigability": "unobstructed",
            "confidence": 0.95,
        }

    # Gera a análise fake de uma região: usa o hook injetado se houver, senão
    # deriva um label determinístico a partir da cor dominante da view de
    # foreground. Emite o mesmo contract que o adapter real
    # (category/label/kind/confidence/alternatives), para que a suíte exercite
    # o schema de produção e não um formato que só o fake fala.
    #
    # O fake lê deliberadamente a view de foreground, e não a de contexto: é
    # assim que a suíte detecta uma regressão em que o raciocínio volte a
    # descrever o entorno em vez da região (#203).
    def analyze_region(
        self, request: RegionReasoningRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        if self._region_response_fn is not None:
            return self._region_response_fn(request)
        foreground = request.foreground_views[0]
        mean_rgb = foreground.payload.pixels.astype(np.float64).mean(axis=(0, 1))
        label = _bucket_label(mean_rgb)
        return {
            "category": "object",
            "label": label,
            "kind": "thing",
            "confidence": 0.9,
            "alternatives": [],
            "description": f"Region with dominant color bucket {label}.",
            "attributes": [],
            "condition": "intact",
            "material": "unknown",
        }

    # Gera a resposta fake de relação entre duas regiões (#206): usa o hook
    # injetado se houver, senão devolve um predicado determinístico derivado do
    # que a geometria já mediu. O default responde ``none`` para pares apenas
    # encostados, porque é isso que a maioria dos pares realmente é: um fake que
    # inventasse relação para todo par tornaria invisível qualquer regressão em
    # que o modelo real passasse a fazer o mesmo.
    def analyze_relation(
        self, request: RegionRelationRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        if self._relation_response_fn is not None:
            return self._relation_response_fn(request)
        if "covers 100%" in request.geometric_summary or "covers 9" in request.geometric_summary:
            return {"predicate": "part_of", "confidence": 0.6}
        return {"predicate": "none"}

    # Levanta BackendExecutionError se region_id estiver na lista configurada
    # para simular falha; usada pelos testes para exercitar o caminho de
    # tratamento de erro do pipeline sem precisar de um backend real falhando.
    def fail_if_configured(self, region_id: str) -> None:
        if region_id in self._fail_on_region_ids:
            raise BackendExecutionError(f"Simulated backend failure for region {region_id!r}.")


# Converte a cor média de um recorte em um label determinístico baseado no
# canal RGB dominante; usado por analyze_region como heurística de conteúdo.
def _bucket_label(mean_rgb: np.ndarray) -> str:
    channel = int(np.argmax(mean_rgb))
    return {0: "reddish_object", 1: "greenish_object", 2: "bluish_object"}[channel]
