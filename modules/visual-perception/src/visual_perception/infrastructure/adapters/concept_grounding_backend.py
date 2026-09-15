"""Adapter real de grounding por conceito com SAM3 PCS (#277).

O SAM3 só segmenta conceitos concretos: prompts genéricos como ``all visible objects``
devolvem zero máscaras (medido em 2026-09-15). Por isso ele recebe os conceitos da
descoberta de cena, e o encoder de imagem roda uma vez por frame: medido na CPU, 16,8 s
para o encoder e ~1,9 s por conceito reaproveitando as features, contra 19,7 s por
conceito com o forward completo, com máscaras e scores idênticos.
"""

from __future__ import annotations

from typing import Any

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import ConceptGroundingConfig
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
from visual_perception.infrastructure.adapters.region_discovery_backend import _mask_to_numpy


# Implementa ConceptRegionDiscoverer com Sam3Model/Sam3Processor, mantendo modelo e
# pós-processamento locais ao adapter.
class Sam3ConceptGroundingAdapter:
    """Satisfaz :class:`~visual_perception.ports.concept_grounding.ConceptRegionDiscoverer` com SAM3 PCS."""

    # Recebe o lifecycle compartilhado para residência e eviction sob pressão de VRAM.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar pesos."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Codifica a imagem uma vez e consulta cada conceito sobre as mesmas features.
    def discover_concept_regions(
        self, image: ImagePayload, concepts: tuple[str, ...], config: ConceptGroundingConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna as propostas de cada conceito, com o conceito na proveniência."""
        if not concepts:
            return ()
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)
        model, processor = self._get_model(config, torch, device)
        key = self._key(config, device)
        pil = payload_to_pil(image, config.backend)
        try:
            image_inputs = processor(images=pil, return_tensors="pt").to(device)
            original_sizes = image_inputs.get("original_sizes").tolist()

            def encode() -> Any:
                with torch.inference_mode():
                    return model.get_vision_features(pixel_values=image_inputs["pixel_values"])

            vision_embeds = self._lifecycle.call_with_eviction(key, encode)
            proposals: list[LocalRegionProposal] = []
            for index, concept in enumerate(concepts):
                text_inputs = processor(text=concept, return_tensors="pt").to(device)

                def ground() -> Any:
                    with torch.inference_mode():
                        outputs = model(
                            vision_embeds=vision_embeds,
                            input_ids=text_inputs["input_ids"],
                            attention_mask=text_inputs["attention_mask"],
                        )
                    return processor.post_process_instance_segmentation(
                        outputs,
                        threshold=config.score_threshold,
                        mask_threshold=config.mask_threshold,
                        target_sizes=original_sizes,
                    )[0]

                result = self._lifecycle.call_with_eviction(key, ground)
                proposals.extend(concept_proposals(result["masks"], result["scores"], image, concept, index, config))
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "o grounding por conceito", error)
        return tuple(proposals)

    # Carrega modelo e processor juntos, residentes sob a mesma chave.
    def _get_model(self, config: ConceptGroundingConfig, torch: Any, device: str) -> tuple[Any, Any]:
        """Retorna o modelo e o processor SAM3 PCS residentes."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)

        def factory() -> tuple[Any, Any]:
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
                raise_backend_execution_error(config.backend, "o carregamento do SAM3 PCS", error)

        return self._lifecycle.get_or_load(self._key(config, device), factory)

    # Chave residente distinta do tracker de discovery, que carrega outra classe.
    def _key(self, config: ConceptGroundingConfig, device: str) -> str:
        """Retorna a chave de residência do modelo de grounding."""
        return f"concept_grounding:{config.backend}:{config.checkpoint}:{device}"


# Converte masks e scores de um conceito em propostas locais ordenadas e limitadas.
# Pública para testar a política de área, score e teto sem GPU.
def concept_proposals(
    masks: Any,
    scores: Any,
    image: ImagePayload,
    concept: str,
    concept_index: int,
    config: ConceptGroundingConfig,
) -> tuple[LocalRegionProposal, ...]:
    """Retorna as propostas válidas de um conceito, em ordem decrescente de score.

    Argumentos:
        masks: máscaras devolvidas pelo pós-processamento.
        scores: scores correspondentes.
        image: imagem que definiu a resolução das máscaras.
        concept: conceito consultado.
        concept_index: posição do conceito na consulta, para ids estáveis.
        config: área mínima, score e teto por conceito.
    Retorna:
        até ``max_regions_per_concept`` propostas com ``concept`` preenchido.
    """
    order = sorted(range(len(masks)), key=lambda index: -float(scores[index]))
    proposals: list[LocalRegionProposal] = []
    for index in order:
        if len(proposals) >= config.max_regions_per_concept:
            break
        score = min(1.0, max(0.0, float(scores[index])))
        segmentation = _mask_to_numpy(masks[index])
        if segmentation.shape != (image.height, image.width):
            raise BackendExecutionError("O SAM3 PCS retornou uma máscara com resolução diferente da imagem.")
        mask = Mask(segmentation, image.width, image.height)
        if score < config.score_threshold or mask.is_empty or mask.area() < config.min_mask_area:
            continue
        proposals.append(
            LocalRegionProposal(
                local_id=f"concept{concept_index}-{len(proposals)}",
                mask=mask,
                box=mask.bounding_box(),
                geometric_confidence=score,
                source=f"sam3-pcs:{config.checkpoint}",
                concept=concept,
            )
        )
    return tuple(proposals)
