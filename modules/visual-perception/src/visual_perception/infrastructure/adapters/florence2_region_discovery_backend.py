"""Adapter do Florence-2 para propostas geométricas de regiões.

O Florence-2 executa a tarefa nativa ``<REGION_PROPOSAL>`` e devolve caixas
sem máscara ou score calibrado. Este adapter transforma cada caixa em uma
máscara retangular para satisfazer o contract de ``RegionDiscoverer``. Ele não
é um ``MultimodalReasoner``: o modelo não aceita os prompts JSON multi-view
usados pelos estágios de contexto, semântica e relações.
"""

from __future__ import annotations

from math import ceil, floor, isfinite
from typing import Any

import numpy as np

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import RegionDiscoveryConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import LocalRegionProposal
from visual_perception.infrastructure.adapters._runtime import (
    payload_to_pil,
    raise_backend_execution_error,
    require_checkpoint,
    require_module,
    resolve_device,
)

FLORENCE2_REGION_PROPOSAL_TASK = "<REGION_PROPOSAL>"


# Implementa RegionDiscoverer com a tarefa nativa de proposta de regiões do
# Florence-2. Existe para expor o modelo na capability que seu output realmente
# satisfaz, sem adaptar artificialmente suas captions ao contract de semântica.
class Florence2RegionDiscoveryAdapter:
    """Satisfaz o port ``RegionDiscoverer`` usando ``<REGION_PROPOSAL>``.

    As propostas são retângulos, pois Florence-2-large não devolve máscaras.
    A geometria fica explicitamente menos precisa que a do SAM e segue pelo
    mesmo merge, filtro e VLM semântico posteriores.
    """

    # Mantém o runtime residente entre tiles e frames, compartilhando o
    # lifecycle com os demais adapters reais compostos pela factory.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem importar runtime nem carregar checkpoint."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Executa a tarefa de proposal na resolução local recebida. É chamada uma
    # vez por tile e preserva o frame local para o remapeamento canônico.
    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna propostas retangulares locais do Florence-2.

        Argumentos:
            image: imagem RGB local do tile atual.
            config: configuração do backend Florence-2 e seus limites.
        Retorna:
            propostas geométricas ordenadas na ordem emitida pelo modelo.
        Levanta:
            BackendUnavailableError: quando checkpoint, dependência ou device não estão disponíveis.
            BackendExecutionError: quando o runtime não consegue carregar ou inferir.
        """
        torch, processor, model, device, key = self._get_runtime(config)
        try:
            inputs = processor(
                text=FLORENCE2_REGION_PROPOSAL_TASK,
                images=payload_to_pil(image, config.backend),
                return_tensors="pt",
            )
            inputs = {name: value.to(device) for name, value in inputs.items()}
            # O checkpoint é carregado em fp16 na GPU, mas ids de token devem
            # permanecer inteiros; por isso somente os pixels acompanham o dtype
            # dos pesos, como recomenda o exemplo oficial do modelo.
            if device == "cuda" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)

            def run() -> Any:
                with torch.inference_mode():
                    return model.generate(
                        **inputs,
                        max_new_tokens=config.max_regions * 16,
                        num_beams=3,
                        do_sample=False,
                    )

            generated = self._lifecycle.call_with_eviction(key, run)
            text = processor.batch_decode(generated, skip_special_tokens=False)[0]
            parsed = processor.post_process_generation(
                text,
                task=FLORENCE2_REGION_PROPOSAL_TASK,
                image_size=(image.width, image.height),
            )
            return _proposals_from_florence_output(parsed, image, config)
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a geração de propostas de região", error)

    # Carrega processor e pesos oficiais somente no primeiro uso. Desde
    # transformers>=4.56 o Florence-2 é suportado nativamente (classe
    # ``Florence2ForConditionalGeneration``); o checkpoint ``microsoft/*``
    # antigo exigia ``trust_remote_code=True`` com código que não acompanhou
    # as mudanças de transformers v5 (ver ``florence-community/*`` no Hub).
    def _get_runtime(self, config: RegionDiscoveryConfig) -> tuple[Any, Any, Any, str, str]:
        """Retorna torch, processor, modelo, device e chave residente."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)
        key = f"region_discovery:{config.backend}:{checkpoint}:{device}"

        def factory() -> tuple[Any, Any]:
            try:
                transformers = require_module("transformers", config.backend)
                processor = transformers.AutoProcessor.from_pretrained(checkpoint)
                model_kwargs: dict[str, Any] = {}
                if device == "cuda":
                    model_kwargs["torch_dtype"] = torch.float16
                model = transformers.Florence2ForConditionalGeneration.from_pretrained(
                    checkpoint, **model_kwargs
                )
                model = model.to(device).eval()
                return processor, model
            except BackendUnavailableError:
                raise
            except (MemoryError, torch.cuda.OutOfMemoryError):
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do Florence-2", error)

        processor, model = self._lifecycle.get_or_load(key, factory)
        return torch, processor, model, device, key


# Traduz a resposta oficial de ``post_process_generation`` em propostas do
# domínio. A função pura permite validar clipping, área e proveniência sem
# baixar o checkpoint nem executar Torch.
def _proposals_from_florence_output(
    output: Any, image: ImagePayload, config: RegionDiscoveryConfig
) -> tuple[LocalRegionProposal, ...]:
    """Converte caixas do Florence-2 em propostas retangulares válidas.

    Argumentos:
        output: objeto retornado pelo post-processamento oficial do Florence-2.
        image: imagem que definiu o sistema local de coordenadas.
        config: limites e proveniência da descoberta de regiões.
    Retorna:
        propostas aceitas, na ordem do modelo e limitadas por ``max_regions``.
    """
    if not isinstance(output, dict):
        raise BackendExecutionError("Florence-2 retornou propostas de região em formato inválido.")
    task_output = output.get(FLORENCE2_REGION_PROPOSAL_TASK, {})
    if not isinstance(task_output, dict):
        raise BackendExecutionError("Florence-2 retornou o payload de propostas em formato inválido.")
    boxes = task_output.get("bboxes", ())
    if not isinstance(boxes, (list, tuple)):
        raise BackendExecutionError("Florence-2 retornou 'bboxes' em formato inválido.")

    proposals: list[LocalRegionProposal] = []
    for box_index, coordinates in enumerate(boxes):
        if len(proposals) >= config.max_regions:
            break
        box = _canonical_box(coordinates, image)
        if box is None:
            continue
        mask = _rectangular_mask(box, image)
        if mask.area() < config.min_mask_area:
            continue
        # Florence-2 não devolve score para REGION_PROPOSAL. ``1.0`` registra
        # somente que o decoder produziu a proposal; não é confiança calibrada.
        proposals.append(
            LocalRegionProposal(
                local_id=f"florence2-{box_index}",
                mask=mask,
                box=box,
                geometric_confidence=1.0,
                source=f"florence2:{config.checkpoint}",
            )
        )
    return tuple(proposals)


# Normaliza uma caixa prevista para a convenção semiaberta do domínio e a
# limita à imagem local, descartando coordenadas malformadas ou degeneradas.
def _canonical_box(coordinates: Any, image: ImagePayload) -> BoundingBox | None:
    """Converte coordenadas Florence-2 em uma caixa canônica ou ``None``."""
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 4:
        return None
    try:
        x_min, y_min, x_max, y_max = (float(value) for value in coordinates)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(value) for value in (x_min, y_min, x_max, y_max)):
        return None
    if x_max <= x_min or y_max <= y_min:
        return None
    return BoundingBox(floor(x_min), floor(y_min), ceil(x_max), ceil(y_max)).clipped_to(
        width=image.width, height=image.height
    )


# Materializa a caixa como mask para atender o contract único de discovery.
# Não tenta inferir contorno: Florence-2 só oferece suporte retangular.
def _rectangular_mask(box: BoundingBox, image: ImagePayload) -> Mask:
    """Cria uma máscara booleana preenchida para uma caixa canônica."""
    mask_data = np.zeros((image.height, image.width), dtype=np.bool_)
    mask_data[int(box.y_min) : int(box.y_max), int(box.x_min) : int(box.x_max)] = True
    return Mask(mask_data, image.width, image.height)
