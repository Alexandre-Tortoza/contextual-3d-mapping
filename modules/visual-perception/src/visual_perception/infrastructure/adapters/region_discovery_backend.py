"""Adapter real de backend de descoberta de regiões.

Issue: #186. Checkpoint selecionado pelo benchmark #174 (SAM ViT-H via
Transformers, ver ``benchmarks/results/benchmark-174-region_discovery-*.json``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal
from visual_perception.infrastructure.adapters._runtime import (
    payload_to_pil,
    raise_backend_execution_error,
    require_checkpoint,
    require_module,
    resolve_device,
)


# Implementa o port RegionDiscoverer com SAM (automatic mask generation via
# Transformers), mantendo detalhes do runtime e do checkpoint locais ao adapter.
class RealRegionDiscoveryAdapter:
    """Satisfaz :class:`~visual_perception.ports.region_discovery.RegionDiscoverer`.

    Usa o pipeline ``mask-generation`` do Transformers (SAM/SAM2, conforme o
    checkpoint configurado) para retornar somente geometria. A classe não
    atribui rótulos nem executa merge: essas responsabilidades continuam nos
    stages canônicos de semântica e de merge.
    """

    # Recebe (ou cria, se omitido) o lifecycle manager que carrega/libera o
    # pipeline sob demanda. Compartilhar o mesmo manager entre os 4 adapters
    # reais (ver ``factory.py``) garante que no máximo um modelo pesado
    # fica residente por vez, mesmo com ``PerceptionPorts`` construído com
    # os 4 adapters já instanciados.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar pesos ou bibliotecas opcionais."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Descobre máscaras class-agnostic na resolução local da imagem recebida.
    # É chamada uma vez por tile pelo estágio de region discovery.
    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna propostas locais do SAM preservando máscaras e confiança geométrica."""
        generator = self._get_generator(config)
        try:
            output = generator(payload_to_pil(image, config.backend), points_per_batch=64)
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a geração automática de máscaras", error)

        masks = output["masks"]
        scores = output["scores"]
        order = sorted(range(len(masks)), key=lambda i: -float(scores[i]))

        proposals: list[LocalRegionProposal] = []
        for rank, index in enumerate(order):
            if len(proposals) >= config.max_regions:
                break
            proposal = _proposal_from_mask(masks[index], float(scores[index]), rank, image, config)
            if proposal is not None:
                proposals.append(proposal)
        return tuple(proposals)

    # Carrega o pipeline SAM apenas quando a configuração real for usada,
    # mantendo a instalação fake-only independente de torch, CUDA e checkpoints.
    def _get_generator(self, config: RegionDiscoveryConfig) -> Any:
        """Retorna o pipeline de mask-generation correspondente à configuração solicitada."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)

        def factory() -> Any:
            try:
                transformers = require_module("transformers", config.backend)
                device_index = 0 if device == "cuda" else -1
                return transformers.pipeline(
                    "mask-generation", model=checkpoint, device=device_index, dtype=torch.float32
                )
            except BackendUnavailableError:
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do SAM", error)

        return self._lifecycle.get_or_load(f"region_discovery:{checkpoint}:{device}", factory)


# Converte uma máscara+score do pipeline de mask-generation no contract local
# do módulo. A box é sempre recalculada pela máscara para preservar a
# convenção semiaberta.
def _proposal_from_mask(
    mask_array: Any,
    score: float,
    index: int,
    image: ImagePayload,
    config: RegionDiscoveryConfig,
) -> LocalRegionProposal | None:
    """Converte uma máscara+score do SAM em proposta local ou descarta saída inválida."""
    segmentation = np.asarray(mask_array, dtype=np.bool_)
    if segmentation.shape != (image.height, image.width):
        raise BackendExecutionError(
            "O SAM retornou uma máscara com resolução diferente da imagem de entrada."
        )
    mask = Mask(segmentation, image.width, image.height)
    if mask.is_empty or mask.area() < config.min_mask_area:
        return None
    confidence = min(1.0, max(0.0, score))
    if confidence < config.score_threshold:
        return None
    return LocalRegionProposal(
        local_id=f"sam-{index}",
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=confidence,
        source=f"sam:{config.checkpoint}",
    )
