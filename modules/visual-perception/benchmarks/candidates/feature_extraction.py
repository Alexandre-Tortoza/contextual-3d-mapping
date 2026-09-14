"""Candidatos de dense visual feature para o benchmark da #174.

Proxy de qualidade: coerência espacial do primeiro componente PCA da grade de
patch-tokens de um frame — o fenômeno bem documentado de que "PC1 de patch
features do DINO segmenta o objeto principal em primeiro plano" (aplicando
threshold no PC1 pela mediana e medindo que fração do lado alto forma um único
blob conectado, em vez de ruído espalhado). Isso não precisa de labels
ground-truth, que este dataset não tem. Uma versão anterior desse proxy usava
a variância explicada total dos top-k componentes, mas isso favorecia
mecanicamente backbones de menor capacidade (mais dimensões ocultas espalham a
variância por mais componentes quase por construção, independente da
qualidade das features); a coerência por connected-component do PC1 não tem
esse viés.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

Candidate = tuple[str, Callable[[], object], Callable[[object], float]]


# Gera um iterador infinito que percorre ciclicamente os frames de referência,
# decodificando cada imagem como RGB sob demanda. Usada pelos candidatos deste
# módulo para alimentar run_once com uma nova imagem a cada chamada, sem
# carregar todos os frames em memória de uma vez.
def _frame_cycle(frames: Iterable[Path]) -> Iterable[Image.Image]:
    for path in itertools.cycle(list(frames)):
        yield Image.open(path).convert("RGB")


# Calcula a coerência espacial do primeiro componente principal da grade de
# patch-tokens: que fração de uma metade da grade, separada pela mediana do
# componente, forma um único blob conectado. É o proxy de qualidade usado por
# run_once para pontuar extração densa sem labels. O sinal de um componente
# principal é arbitrário — ``(u, s, v)`` e ``(-u, s, -v)`` são decomposições
# igualmente válidas —, então a métrica mede as duas metades e fica com a mais
# coerente: antes ela media o objeto para um backbone e o fundo para outro.
def first_component_coherence(patch_tokens: np.ndarray, grid_height: int, grid_width: int) -> float:
    """Retorna a coerência espacial do primeiro componente principal, independente do sinal.

    Argumentos:
        patch_tokens: tokens de patch ``(grid_height * grid_width, dimensão)``.
        grid_height: linhas da grade de patches.
        grid_width: colunas da grade de patches.
    Retorna:
        a maior fração ocupada por um único componente conectado, entre as duas
        metades da grade.
    """
    centered = patch_tokens - patch_tokens.mean(axis=0, keepdims=True)
    left_singular_vectors, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    component = (left_singular_vectors[:, 0] * singular_values[0]).reshape(grid_height, grid_width)
    median = np.median(component)
    return max(_largest_blob_fraction(component > median), _largest_blob_fraction(component < median))


# Fração de uma máscara ocupada pelo seu maior componente conectado.
def _largest_blob_fraction(mask: np.ndarray) -> float:
    """Retorna a fração da máscara coberta pelo maior componente conectado, ou 0."""
    if not mask.any():
        return 0.0
    labeled, component_count = ndimage.label(mask)
    component_sizes = ndimage.sum(mask, labeled, index=range(1, component_count + 1))
    return float(np.max(component_sizes) / mask.sum())


# Constrói um candidato de benchmark (nome, factory, run_once) para um
# checkpoint da família DINOv2: factory carrega o processor/modelo na GPU, e
# run_once extrai os patch-tokens de um frame e retorna a coerência do primeiro componente principal
# (via first_component_coherence) como score de qualidade. Chamada por candidates() para
# montar a lista de candidatos comparados por backend_benchmark.py.
def _dinov2_candidate(name: str, checkpoint: str, frames: list[Path]) -> Candidate:
    frame_iter = _frame_cycle(frames)

    # Carrega o processor e o modelo DINOv2 do checkpoint indicado, movendo o
    # modelo para a GPU em modo eval; chamada uma única vez por
    # benchmark_candidate antes das rodadas de warmup/medição.
    def factory() -> object:
        from transformers import AutoImageProcessor, Dinov2Model

        processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = Dinov2Model.from_pretrained(checkpoint, dtype=torch.float32).to("cuda").eval()
        return (processor, model)

    # Roda uma inferência do DINOv2 sobre o próximo frame do ciclo e reduz os
    # patch-tokens resultantes à métrica de coerência do primeiro componente principal; é o run_once
    # exigido pelo contrato de benchmark_candidate.
    def run_once(bundle: object) -> float:
        processor, model = bundle  # type: ignore[misc]
        image = next(frame_iter)
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        register_token_count = getattr(model.config, "num_register_tokens", 0)
        patch_tokens = outputs.last_hidden_state[0, 1 + register_token_count :, :]
        patch_size = model.config.patch_size
        grid_height = inputs["pixel_values"].shape[-2] // patch_size
        grid_width = inputs["pixel_values"].shape[-1] // patch_size
        return first_component_coherence(patch_tokens.float().cpu().numpy(), grid_height, grid_width)

    return (name, factory, run_once)


# Monta a lista de candidatos de dense feature extraction avaliados pelo
# benchmark da #174: as duas variantes de checkpoint do DINOv2 (base e large).
# Chamada por run_backend_benchmark.py ao selecionar o stage
# "feature_extraction".
def candidates(frames: list[Path]) -> list[Candidate]:
    return [
        _dinov2_candidate("dinov2:facebook/dinov2-base", "facebook/dinov2-base", frames),
        _dinov2_candidate("dinov2:facebook/dinov2-large", "facebook/dinov2-large", frames),
    ]
