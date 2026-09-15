"""Adapters reais de backend de descoberta de regiões.

SAM, SAM2 e SAM3 produzem propostas por geração automática de máscaras
(*segment everything*): uma grade de pontos percorre a imagem e o modelo
devolve máscaras class-agnostic. No SAM3 a geração usa o tracker, a interface
de prompts por ponto do modelo; o prompt textual (PCS) não é usado porque
frases genéricas não produzem máscaras.
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


# Implementa o port RegionDiscoverer com geração automática de máscaras via
# Transformers, mantendo detalhes do runtime e do checkpoint locais ao adapter.
class RealRegionDiscoveryAdapter:
    """Satisfaz :class:`~visual_perception.ports.region_discovery.RegionDiscoverer`.

    Usa o pipeline ``mask-generation`` do Transformers com SAM/SAM2 (backend
    ``sam``) ou com o tracker do SAM3 (backend ``sam3``) para retornar somente
    geometria. A classe não atribui rótulos nem executa merge: essas
    responsabilidades continuam nos stages canônicos de semântica e de merge.
    """

    # Recebe (ou cria, se omitido) o lifecycle manager que carrega e mantém
    # residente o pipeline sob demanda. Compartilhar o mesmo manager entre
    # os 4 adapters reais (ver ``factory.py``) permite reaproveitar modelos
    # já residentes entre estágios intercalados no mesmo frame.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar pesos ou bibliotecas opcionais."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Descobre máscaras class-agnostic na resolução local da imagem recebida.
    # É chamada uma vez por tile pelo estágio de region discovery.
    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna propostas locais preservando máscaras e confiança geométrica."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)
        key = f"region_discovery:{config.backend}:{checkpoint}:{device}"
        generator = self._get_generator(config)
        try:
            output = self._lifecycle.call_with_eviction(
                key,
                lambda: generator(
                    payload_to_pil(image, config.backend),
                    points_per_batch=16,
                    pred_iou_thresh=config.pred_iou_threshold,
                    stability_score_thresh=config.stability_score_threshold,
                ),
            )
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a geração automática de máscaras", error)

        return _proposals_from_mask_outputs(output["masks"], output["scores"], image, config)

    # Carrega o pipeline apenas quando a configuração real for usada, mantendo
    # a instalação fake-only independente de torch, CUDA e checkpoints.
    def _get_generator(self, config: RegionDiscoveryConfig) -> Any:
        """Retorna o pipeline de mask-generation correspondente à configuração solicitada."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)

        def factory() -> Any:
            try:
                transformers = require_module("transformers", config.backend)
                device_index = 0 if device == "cuda" else -1
                if config.backend == "sam3":
                    # O checkpoint do SAM3 não é resolvido pelo AutoModel de
                    # mask-generation; o tracker e seu image processor são
                    # entregues explicitamente ao pipeline.
                    tracker = transformers.Sam3TrackerModel.from_pretrained(checkpoint).to(device).eval()
                    processor = transformers.Sam3TrackerProcessor.from_pretrained(checkpoint)
                    return transformers.pipeline(
                        "mask-generation",
                        model=tracker,
                        image_processor=processor.image_processor,
                        device=device_index,
                    )
                return transformers.pipeline(
                    "mask-generation", model=checkpoint, device=device_index, dtype=torch.float32
                )
            except BackendUnavailableError:
                raise
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # Ver comentário equivalente em feature_extraction_backend.py:
                # preserva o tipo de OOM para que _load_with_eviction possa
                # liberar residentes por LRU e tentar de novo.
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do gerador de máscaras", error)

        return self._lifecycle.get_or_load(f"region_discovery:{config.backend}:{checkpoint}:{device}", factory)


# Ordena e limita a saída de masks do runtime antes de traduzi-la ao contract
# local. Separada de ``discover`` para que a política de score, área e máximo
# de propostas seja testável sem GPU nem checkpoint.
def _proposals_from_mask_outputs(
    masks: Any,
    scores: Any,
    image: ImagePayload,
    config: RegionDiscoveryConfig,
) -> tuple[LocalRegionProposal, ...]:
    """Converte a saída de masks do runtime em propostas locais ordenadas.

    Argumentos:
        masks: masks binárias ou tensores devolvidos pelo runtime.
        scores: confidences geométricas correspondentes às masks.
        image: imagem local do tile que definiu as dimensões das masks.
        config: backend, limites e thresholds de region discovery.
    Retorna:
        propostas válidas, em ordem decrescente de score.
    """
    order = sorted(range(len(masks)), key=lambda index: -float(scores[index]))
    proposals: list[LocalRegionProposal] = []
    for rank, index in enumerate(order):
        if len(proposals) >= config.max_regions:
            break
        proposal = _proposal_from_mask(masks[index], float(scores[index]), rank, image, config)
        if proposal is not None:
            proposals.append(proposal)
    return tuple(proposals)


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
    segmentation = _mask_to_numpy(mask_array)
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
        local_id=f"{config.backend}-{index}",
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=confidence,
        source=f"{config.backend}:{config.checkpoint}",
    )


# Converte tanto arrays quanto tensores do Transformers para a máscara booleana
# que o domínio entende, sem fazer torch parte da fronteira pública do módulo.
def _mask_to_numpy(mask_array: Any) -> np.ndarray:
    """Converte uma máscara de runtime em array booleano de NumPy."""
    if hasattr(mask_array, "detach"):
        mask_array = mask_array.detach().cpu().numpy()
    return np.asarray(mask_array, dtype=np.bool_)
