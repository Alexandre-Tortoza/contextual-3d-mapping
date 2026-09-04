"""Fronteira substituível de descoberta de regiões.

Issue: #158.

Implementações descobrem regiões da imagem de forma independente do
raciocínio semântico: elas retornam apenas geometria e uma confiança
geométrica, nunca um rótulo semântico. Essa fronteira precisa continuar
satisfazível por um fake sem GPU (ver
``infrastructure/fakes/fake_region_discoverer.py``) e nunca deve vazar tipos
de tensor/modelo específicos de backend.
"""

from __future__ import annotations

from typing import Protocol

from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal


# Port que desacopla o pipeline do backend concreto de descoberta de regiões
# (ex: SAM, um detector de propostas). Existe para manter geometria separada
# de semântica: quem atribui rótulos é o estágio de region semantics, não
# este port.
class RegionDiscoverer(Protocol):
    """Descobre propostas de região class-agnostic, promptable, ou densas."""

    # Ponto de entrada único do port: recebe uma imagem e devolve as
    # propostas de região encontradas. Chamado pelo estágio de region
    # discovery do pipeline, antes de qualquer merge ou rotulagem semântica.
    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna zero, uma ou várias propostas em coordenadas locais de ``image``."""
        ...
