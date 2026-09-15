"""Adapters reais de backend de descoberta de regiões.

SAM/SAM2 produzem propostas automaticamente. SAM3 usa um prompt amplo
versionado para produzir propostas geométricas sem publicar semântica.
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
        """Retorna propostas locais do SAM preservando máscaras e confiança geométrica."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)
        key = f"region_discovery:{checkpoint}:{device}"
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
            except (MemoryError, torch.cuda.OutOfMemoryError):
                # Ver comentário equivalente em feature_extraction_backend.py:
                # preserva o tipo de OOM para que _load_with_eviction possa
                # liberar residentes por LRU e tentar de novo.
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do SAM", error)

        return self._lifecycle.get_or_load(f"region_discovery:{checkpoint}:{device}", factory)


# Implementa RegionDiscoverer com SAM3, cujo runtime exige um prompt textual.
# O prompt amplo pertence à config para que a cobertura produzida seja
# reprodutível e participe do fingerprint do stage.
class Sam3RegionDiscoveryAdapter:
    """Satisfaz ``RegionDiscoverer`` usando segmentação condicionada por texto do SAM3."""

    # Recebe o lifecycle compartilhado para manter o modelo SAM3 residente entre
    # tiles e delegar a eviction sob pressão de VRAM ao dono central.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar pesos ou módulos opcionais."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Segmenta todas as instâncias que correspondem ao prompt amplo da config.
    # É chamado por tile pelo stage de discovery antes de qualquer semântica.
    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna propostas locais do SAM3 preservando máscaras e confiança geométrica."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)
        key = f"region_discovery:sam3:{checkpoint}:{device}"
        model, processor = self._get_model(config, torch, device)
        pil = payload_to_pil(image, config.backend)
        try:
            # Mantém forward e pós-processamento na mesma chamada gerenciada,
            # pois ambos retêm intermediários CUDA que podem exigir eviction.
            def infer() -> Any:
                """Executa o forward e restaura as máscaras nas dimensões da imagem."""
                inputs = processor(images=pil, text=config.prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                return processor.post_process_instance_segmentation(
                    outputs,
                    threshold=config.score_threshold,
                    mask_threshold=config.mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist(),
                )[0]

            output = self._lifecycle.call_with_eviction(key, infer)
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a segmentação pelo prompt amplo", error)

        masks = output["masks"]
        scores = output["scores"]
        order = sorted(range(len(masks)), key=lambda index: -float(scores[index]))
        proposals: list[LocalRegionProposal] = []
        for rank, index in enumerate(order):
            if len(proposals) >= config.max_regions:
                break
            proposal = _proposal_from_mask(
                masks[index], float(scores[index]), rank, image, config, backend_name="sam3"
            )
            if proposal is not None:
                proposals.append(proposal)
        return tuple(proposals)

    # Carrega modelo e processor juntos porque ambos são necessários para o
    # contract do checkpoint e precisam compartilhar a mesma residência.
    def _get_model(self, config: RegionDiscoveryConfig, torch: Any, device: str) -> tuple[Any, Any]:
        """Retorna o modelo e processor SAM3 residentes para a configuração solicitada."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)

        def factory() -> tuple[Any, Any]:
            """Carrega SAM3 por meio das classes públicas do Transformers."""
            try:
                transformers = require_module("transformers", config.backend)
                return (
                    transformers.Sam3Model.from_pretrained(checkpoint).to(device).eval(),
                    transformers.Sam3Processor.from_pretrained(checkpoint),
                )
            except BackendUnavailableError:
                raise
            except (MemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do SAM3", error)

        key = f"region_discovery:sam3:{checkpoint}:{device}"
        return self._lifecycle.get_or_load(key, factory)


# Converte uma máscara+score do pipeline de mask-generation no contract local
# do módulo. A box é sempre recalculada pela máscara para preservar a
# convenção semiaberta.
def _proposal_from_mask(
    mask_array: Any,
    score: float,
    index: int,
    image: ImagePayload,
    config: RegionDiscoveryConfig,
    backend_name: str = "sam",
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
        local_id=f"{backend_name}-{index}",
        mask=mask,
        box=mask.bounding_box(),
        geometric_confidence=confidence,
        source=(
            f"sam3:{config.checkpoint}:prompt={config.prompt}"
            if backend_name == "sam3"
            else f"sam:{config.checkpoint}"
        ),
    )


# Converte tanto arrays quanto tensores do Transformers para a máscara booleana
# que o domínio entende, sem fazer torch parte da fronteira pública do módulo.
def _mask_to_numpy(mask_array: Any) -> np.ndarray:
    """Converte uma máscara de runtime em array booleano de NumPy."""
    if hasattr(mask_array, "detach"):
        mask_array = mask_array.detach().cpu().numpy()
    return np.asarray(mask_array, dtype=np.bool_)
