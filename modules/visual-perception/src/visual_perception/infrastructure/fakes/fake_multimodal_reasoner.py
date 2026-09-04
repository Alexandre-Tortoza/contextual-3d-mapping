"""Fake determinístico e livre de GPU para a fronteira de raciocínio multimodal (#164, #165, #189)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendExecutionError
from visual_perception.domain.image_payload import ImagePayload

SceneResponseFn = Callable[[ImagePayload], dict[str, Any]]
RegionResponseFn = Callable[[ImagePayload, "str | None"], dict[str, Any]]


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
    ) -> None:
        self._scene_response_fn = scene_response_fn
        self._region_response_fn = region_response_fn
        self._fail_on_region_ids = fail_on_region_ids

    # Gera a análise fake de cena completa: usa o hook injetado se houver,
    # senão deriva um resultado determinístico a partir do brilho médio da
    # imagem.
    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> dict[str, Any]:
        if self._scene_response_fn is not None:
            return self._scene_response_fn(image)
        brightness = float(image.pixels.mean())
        return {
            "scene_type": "well_lit" if brightness > 127 else "dim",
            "description": f"Synthetic scene with mean brightness {brightness:.1f}.",
            "attributes": ["indoor"],
            "hazards": [],
            "confidence": 0.95,
        }

    # Gera a análise fake de uma região/recorte: usa o hook injetado se
    # houver, senão deriva um label determinístico a partir da cor
    # dominante do recorte.
    def analyze_region(
        self,
        image: ImagePayload,
        mask_crop: ImagePayload,
        scene_summary: str | None,
        config: MultimodalReasoningConfig,
    ) -> dict[str, Any]:
        if self._region_response_fn is not None:
            return self._region_response_fn(mask_crop, scene_summary)
        mean_rgb = mask_crop.pixels.astype(np.float64).mean(axis=(0, 1))
        label = _bucket_label(mean_rgb)
        return {
            "labels": [{"value": label, "confidence": 0.9}],
            "description": f"Region with dominant color bucket {label}.",
            "attributes": [],
            "condition": "intact",
            "material": "unknown",
        }

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
