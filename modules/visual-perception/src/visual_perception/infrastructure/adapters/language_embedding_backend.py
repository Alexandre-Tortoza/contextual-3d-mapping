"""Adapter real de backend de embedding alinhado a linguagem.

Issue: #188. Checkpoint selecionado pelo benchmark #174 (CLIP ViT-L/14 via
Transformers, ver ``benchmarks/results/benchmark-174-language_embedding-*.json``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import LanguageEmbeddingConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.infrastructure.adapters._runtime import (
    payload_to_pil,
    raise_backend_execution_error,
    require_checkpoint,
    require_module,
    resolve_device,
)


# Implementa o port LanguageAlignedEncoder com CLIP (Transformers) e preserva
# o espaço compartilhado de embeddings para imagem e texto.
class RealLanguageAlignedEncoderAdapter:
    """Satisfaz :class:`~visual_perception.ports.language_embedding.LanguageAlignedEncoder`.

    Usa CLIP via Transformers para garantir que imagem e texto compartilhem o
    mesmo espaço de embedding, sem expor tensors para os consumidores do port.
    """

    # Recebe (ou cria, se omitido) o lifecycle manager que carrega/libera o
    # processor/modelo sob demanda. Compartilhar o mesmo manager entre os 4
    # adapters reais (ver ``factory.py``) garante que no máximo um modelo
    # pesado fica residente por vez.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem carregar pesos ou dependências opcionais."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Codifica um crop RGB no espaço compartilhado de embedding. É chamado
    # pelo estágio de language embedding para cada região canônica.
    def encode_image(self, image: ImagePayload, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        """Retorna o embedding CLIP normalizado de um crop de imagem."""
        torch, processor, model, device = self._get_runtime(config)
        try:
            inputs = processor(images=payload_to_pil(image, config.backend), return_tensors="pt")
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                vector = _unwrap_features(model.get_image_features(**inputs))
            return _to_vector(vector, config, config.backend)
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "o encoding CLIP de imagem", error)

    # Codifica texto no mesmo espaço que encode_image para suportar consulta
    # cross-modal sem converter vetores em outro estágio do módulo.
    def encode_text(self, text: str, config: LanguageEmbeddingConfig) -> tuple[float, ...]:
        """Retorna o embedding CLIP normalizado de uma query textual."""
        if not text:
            raise ValueError("O texto para embedding não pode estar vazio.")
        torch, processor, model, device = self._get_runtime(config)
        try:
            inputs = processor(text=[text], return_tensors="pt", padding=True)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                vector = _unwrap_features(model.get_text_features(**inputs))
            return _to_vector(vector, config, config.backend)
        except BackendExecutionError:
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "o encoding CLIP de texto", error)

    # Carrega um único modelo CLIP para imagem e texto, delegando residência
    # ao lifecycle manager compartilhado. Essa propriedade é necessária para
    # cumprir o contract de espaço de embedding compartilhado.
    def _get_runtime(self, config: LanguageEmbeddingConfig) -> tuple[Any, Any, Any, str]:
        """Retorna torch, processor, modelo e device para a configuração solicitada."""
        checkpoint = require_checkpoint(config.checkpoint, config.backend)
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)

        def factory() -> tuple[Any, Any]:
            try:
                transformers = require_module("transformers", config.backend)
                processor = transformers.AutoProcessor.from_pretrained(checkpoint)
                model = (
                    transformers.AutoModel.from_pretrained(checkpoint, dtype=torch.float32)
                    .to(device)
                    .eval()
                )
                return processor, model
            except BackendUnavailableError:
                raise
            except Exception as error:
                raise_backend_execution_error(config.backend, "o carregamento do CLIP", error)

        processor, model = self._lifecycle.get_or_load(
            f"language_embedding:{checkpoint}:{device}", factory
        )
        return torch, processor, model, device


# Desembrulha o resultado de get_text_features/get_image_features entre
# versões do Transformers: em versões recentes (5.x) esses métodos retornam
# um objeto ``BaseModelOutputWithPooling`` cujo ``.pooler_output`` guarda o
# embedding projetado, em vez de um tensor puro; versões mais antigas
# retornavam o tensor diretamente.
def _unwrap_features(output: Any) -> Any:
    """Extrai o tensor de embedding de uma saída de get_*_features, com ou sem pooler_output."""
    pooler_output = getattr(output, "pooler_output", None)
    return pooler_output if pooler_output is not None else output


# Converte a saída tensorial do CLIP em vetor finito e validado pelo contract
# do módulo. A normalização L2 é aplicada uma única vez.
def _to_vector(vector: Any, config: LanguageEmbeddingConfig, backend: str) -> tuple[float, ...]:
    """Valida, normaliza e converte um tensor CLIP em tupla de floats."""
    values = vector[0].detach().float().cpu().numpy()
    if values.ndim != 1 or int(values.shape[0]) != config.dimension:
        raise BackendExecutionError(
            f"O backend {backend!r} retornou embedding de dimensão {values.shape}; "
            f"era esperada dimensão {config.dimension}."
        )
    if not np.isfinite(values).all():
        raise BackendExecutionError(f"O backend {backend!r} retornou embedding com NaN ou Inf.")
    if config.normalize:
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            raise BackendExecutionError(f"O backend {backend!r} retornou embedding nulo.")
        values = values / norm
    return tuple(float(value) for value in values)
