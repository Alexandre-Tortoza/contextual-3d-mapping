"""Helpers locais para carregar runtimes opcionais de inferência.

Os adapters reais não podem importar PyTorch, Pillow ou bibliotecas de
modelos durante o import do pacote: o módulo ainda precisa funcionar com os
fakes em máquinas sem GPU. Este arquivo concentra essa política sem vazar
tipos de runtime para domain ou application.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload


# Importa uma dependência opcional apenas no instante de uso do adapter. Isso
# mantém os fakes e a API pública importáveis em ambientes sem extras de ML.
def require_module(module_name: str, backend: str) -> Any:
    """Retorna um módulo opcional ou levanta um diagnóstico acionável."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        raise BackendUnavailableError(
            f"O backend {backend!r} requer o módulo opcional {module_name!r}. "
            "Instale os extras de ML do visual-perception."
        ) from error


# Resolve o device solicitado respeitando a disponibilidade efetiva de CUDA.
# É usado por todos os adapters para terem a mesma semântica de configuração.
def resolve_device(torch: Any, requested_device: str, backend: str) -> str:
    """Resolve ``auto`` e valida que CUDA está disponível quando exigida."""
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested_device == "cuda" and not cuda_available:
        raise BackendUnavailableError(
            f"O backend {backend!r} foi configurado para CUDA, mas nenhuma GPU CUDA está disponível."
        )
    return requested_device


# Converte o payload canônico em imagem PIL sem introduzir Pillow na fronteira
# pública. É usado pelos processors de Transformers e OpenCLIP.
def payload_to_pil(image: ImagePayload, backend: str) -> Any:
    """Converte um ``ImagePayload`` RGB em uma imagem PIL compatível com backends."""
    pil_image = require_module("PIL.Image", backend)
    pixels = np.asarray(image.pixels)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    return pil_image.fromarray(pixels, mode="RGB")


# Padroniza a tradução de falhas de bibliotecas externas para o erro estável
# do módulo. O encadeamento preserva debugging sem expor tipos na API pública.
def raise_backend_execution_error(backend: str, operation: str, error: Exception) -> None:
    """Levanta ``BackendExecutionError`` com contexto do backend e da operação."""
    raise BackendExecutionError(
        f"O backend {backend!r} falhou durante {operation}: {error}"
    ) from error


# Exige um checkpoint explícito para evitar que uma configuração real use o
# placeholder ``none`` e produza uma falha obscura dentro da biblioteca.
def require_checkpoint(checkpoint: str, backend: str) -> str:
    """Valida e retorna o identificador de checkpoint de um backend real."""
    if not checkpoint or checkpoint == "none":
        raise BackendUnavailableError(
            f"O backend {backend!r} exige um checkpoint explícito; o valor atual é {checkpoint!r}."
        )
    return checkpoint
