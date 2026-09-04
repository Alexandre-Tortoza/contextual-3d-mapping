"""Fake determinístico e livre de GPU para a fronteira de descoberta de regiões (#158)."""

from __future__ import annotations

import numpy as np

from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal


# Implementação fake do port RegionDiscoverer. Existe para permitir testar e
# desenvolver o pipeline sem GPU nem modelo real, substituindo o adapter
# real (#186) até que ele esteja pronto.
class FakeRegionDiscoverer:
    """Encontra componentes conectados de pixels que não são background.

    Determinístico e sem dependências: quem chama controla a quantidade de
    regiões controlando o conteúdo da imagem sintética (uma imagem lisa
    produz zero regiões, um blob colorido produz uma, vários produzem
    vários).
    """

    # Guarda os parâmetros que controlam a sensibilidade da detecção: o
    # limiar que separa foreground de background e a área mínima para uma
    # região ser considerada válida.
    def __init__(self, background_threshold: int = 250, min_area: int = 4) -> None:
        self.background_threshold = background_threshold
        self.min_area = min_area

    # Descobre as regiões fake da imagem: segmenta foreground/background por
    # limiar, extrai componentes conectados e converte cada um em uma
    # LocalRegionProposal.
    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        foreground = image.pixels.min(axis=2) < self.background_threshold
        components = _connected_components(foreground)
        proposals = []
        for index, component in enumerate(components):
            if int(component.sum()) < self.min_area:
                continue
            mask = Mask(component, image.width, image.height)
            proposals.append(
                LocalRegionProposal(
                    local_id=f"fake-{index}",
                    mask=mask,
                    box=mask.bounding_box(),
                    geometric_confidence=max(config.score_threshold, 0.6),
                    source="fake_region_discoverer",
                )
            )
        return tuple(proposals)


# Rotula componentes conectados (4-conectividade) na máscara de foreground,
# sem depender de bibliotecas externas de visão computacional; usado por
# discover() para transformar pixels de foreground em regiões distintas.
def _connected_components(foreground: np.ndarray) -> list[np.ndarray]:
    """Rotulagem de componentes conectados com 4-conectividade, sem dependência externa."""
    visited = np.zeros_like(foreground, dtype=np.bool_)
    height, width = foreground.shape
    components: list[np.ndarray] = []
    for start_y in range(height):
        for start_x in range(width):
            if not foreground[start_y, start_x] or visited[start_y, start_x]:
                continue
            component = np.zeros_like(foreground, dtype=np.bool_)
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            while stack:
                y, x = stack.pop()
                component[y, x] = True
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and foreground[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(component)
    components.sort(key=lambda component: int(component.sum()), reverse=True)
    return components
