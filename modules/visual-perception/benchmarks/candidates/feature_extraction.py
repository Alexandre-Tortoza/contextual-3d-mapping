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


# Calcula a coerência espacial (connected-component) do primeiro componente
# PCA (PC1) da grade de patch-tokens: mede que fração da região "acima da
# mediana" do PC1 forma um único blob conectado, em vez de ruído espalhado.
# É o proxy de qualidade usado por run_once para pontuar um candidato de
# dense feature extraction sem precisar de labels ground-truth.
def _pc1_coherence(patch_tokens: np.ndarray, grid_h: int, grid_w: int) -> float:
    centered = patch_tokens - patch_tokens.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    pc1 = (u[:, 0] * singular_values[0]).reshape(grid_h, grid_w)
    mask = pc1 > np.median(pc1)
    if not mask.any():
        return 0.0
    labeled, num_components = ndimage.label(mask)
    if num_components == 0:
        return 0.0
    component_sizes = ndimage.sum(mask, labeled, index=range(1, num_components + 1))
    return float(component_sizes.max() / mask.sum())


# Constrói um candidato de benchmark (nome, factory, run_once) para um
# checkpoint da família DINOv2: factory carrega o processor/modelo na GPU, e
# run_once extrai os patch-tokens de um frame e retorna a coerência do PC1
# (via _pc1_coherence) como score de qualidade. Chamada por candidates() para
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
    # patch-tokens resultantes à métrica de coerência do PC1; é o run_once
    # exigido pelo contrato de benchmark_candidate.
    def run_once(bundle: object) -> float:
        processor, model = bundle  # type: ignore[misc]
        image = next(frame_iter)
        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        num_register_tokens = getattr(model.config, "num_register_tokens", 0)
        patch_tokens = outputs.last_hidden_state[0, 1 + num_register_tokens :, :]
        patch_size = model.config.patch_size
        grid_h = inputs["pixel_values"].shape[-2] // patch_size
        grid_w = inputs["pixel_values"].shape[-1] // patch_size
        return _pc1_coherence(patch_tokens.float().cpu().numpy(), grid_h, grid_w)

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
