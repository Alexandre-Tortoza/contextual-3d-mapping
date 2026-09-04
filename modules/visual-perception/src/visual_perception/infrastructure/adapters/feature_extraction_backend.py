"""Adapter real de backend de features visuais densas.

Issue: #187. A seleção final de checkpoint depende do benchmark #174.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.feature_map import FeatureMap
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.infrastructure.adapters._runtime import (
    payload_to_pil,
    raise_backend_execution_error,
    require_checkpoint,
    require_module,
    resolve_device,
)


# Implementa o port DenseFeatureExtractor com DINOv2, isolando tensors e
# detalhes de Transformers da camada de application.
class RealDenseFeatureExtractionAdapter:
    """Satisfaz :class:`~visual_perception.ports.feature_extraction.DenseFeatureExtractor`.

    Usa DINOv2 via Transformers e expõe um grid espacial sem tokens ou tipos
    privados do modelo.
    """

    # Recebe (ou cria, se omitido) o lifecycle manager que carrega/libera o
    # processor/modelo sob demanda. Compartilhar o mesmo manager entre os 4
    # adapters reais (ver ``factory.py``) garante que no máximo um modelo
    # pesado fica residente por vez.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar o modelo DINOv2."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Extrai tokens espaciais e os reconstitui em um FeatureMap alinhado à
    # imagem de entrada; usado pelo pooling mask-aware do pipeline canônico.
    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        """Retorna um grid denso DINOv2, sem tokens CLS ou register, para a imagem dada."""
        torch, processor, model, device = self._get_runtime(config)
        try:
            inputs = processor(images=payload_to_pil(image, config.backend), return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
            tokens = output.last_hidden_state[0]
            pixel_values = inputs["pixel_values"]
            grid_height, grid_width, prefix_tokens = _feature_grid_shape(
                model, pixel_values, int(tokens.shape[0])
            )
            spatial_tokens = tokens[prefix_tokens:]
            if int(spatial_tokens.shape[0]) != grid_height * grid_width:
                raise BackendExecutionError(
                    "O DINOv2 retornou uma quantidade de tokens incompatível com sua grade espacial."
                )
            data = spatial_tokens.detach().float().cpu().numpy().reshape(
                grid_height, grid_width, int(spatial_tokens.shape[-1])
            )
            return FeatureMap(
                data=np.asarray(data, dtype=np.float64),
                stride_x=image.width / grid_width,
                stride_y=image.height / grid_height,
                dimension=int(data.shape[-1]),
                model_id=config.checkpoint,
            )
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a extração de features DINOv2", error)

    # Carrega processor e modelo sob demanda, delegando residência ao
    # lifecycle manager compartilhado.
    def _get_runtime(self, config: FeatureExtractionConfig) -> tuple[Any, Any, Any, str]:
        """Retorna torch, processor, modelo e device para a configuração solicitada."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)

        def factory() -> tuple[Any, Any]:
            try:
                transformers = require_module("transformers", config.backend)
                processor = transformers.AutoImageProcessor.from_pretrained(checkpoint)
                model = transformers.AutoModel.from_pretrained(checkpoint).to(device).eval()
                return processor, model
            except BackendUnavailableError:
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do DINOv2", error)

        processor, model = self._lifecycle.get_or_load(
            f"feature_extraction:{checkpoint}:{device}", factory
        )
        return torch, processor, model, device


# Determina a grade espacial DINOv2 a partir da resolução efetivamente
# processada, do patch size e dos tokens prefixados pelo modelo.
def _feature_grid_shape(model: Any, pixel_values: Any, token_count: int) -> tuple[int, int, int]:
    """Calcula altura e largura da grade espacial produzida pelo DINOv2."""
    patch_size = model.config.patch_size
    if isinstance(patch_size, tuple):
        patch_height, patch_width = patch_size
    else:
        patch_height = patch_width = int(patch_size)
    grid_height = int(pixel_values.shape[-2]) // patch_height
    grid_width = int(pixel_values.shape[-1]) // patch_width
    register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    prefix_tokens = 1 + register_tokens
    if grid_height <= 0 or grid_width <= 0 or token_count != grid_height * grid_width + prefix_tokens:
        raise BackendExecutionError("Não foi possível reconstruir a grade espacial retornada pelo DINOv2.")
    return grid_height, grid_width, prefix_tokens
