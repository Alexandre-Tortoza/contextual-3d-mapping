"""Candidatos de region discovery para o benchmark da #174.

Cada candidato é uma tripla ``(name, factory, run_once)`` que segue o
contrato de ``backend_benchmark.benchmark_candidate``. ``run_once`` consome um
frame do conjunto representativo corridor-02 por chamada (via um iterador
cíclico compartilhado) e retorna a própria média de IoU previsto pelo modelo
entre as masks que ele gerou para aquele frame — um sinal de qualidade
genuíno, nativo do modelo, por frame, em ``[0, 1]``, não um proxy inventado
depois.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Candidate = tuple[str, Callable[[], object], Callable[[object], float]]


# Gera um iterador infinito que percorre ciclicamente os frames de referência,
# decodificando cada imagem como RGB sob demanda. Usada pelos candidatos deste
# módulo para alimentar run_once com uma nova imagem a cada chamada.
def _frame_cycle(frames: Iterable[Path]) -> Iterable[Image.Image]:
    for path in itertools.cycle(list(frames)):
        yield Image.open(path).convert("RGB")


# Constrói um candidato de benchmark (nome, factory, run_once) para um
# checkpoint da família SAM (via pipeline "mask-generation" do transformers):
# factory carrega o gerador de masks, e run_once roda a geração sobre um frame
# e retorna a média das confidence scores nativas do modelo. Chamada por
# candidates() para montar a lista de candidatos comparados por
# backend_benchmark.py.
def _sam_family_candidate(name: str, checkpoint: str, frames: list[Path]) -> Candidate:
    frame_iter = _frame_cycle(frames)

    # Carrega o pipeline "mask-generation" do transformers para o checkpoint
    # indicado, na GPU; chamada uma única vez por benchmark_candidate.
    def factory() -> object:
        from transformers import pipeline

        return pipeline("mask-generation", model=checkpoint, device=0, dtype=torch.float32)

    # Gera masks para o próximo frame do ciclo e retorna a média das
    # confidence scores retornadas pelo próprio modelo; é o run_once exigido
    # pelo contrato de benchmark_candidate.
    def run_once(generator: object) -> float:
        image = next(frame_iter)
        out = generator(image, points_per_batch=64)  # type: ignore[operator]
        scores = out["scores"]
        if len(scores) == 0:
            return 0.0
        return float(sum(float(s) for s in scores) / len(scores))

    return (name, factory, run_once)


# Constrói um candidato de benchmark (nome fixo "fastsam:<checkpoint>",
# factory, run_once) para um checkpoint FastSAM via ultralytics: factory
# carrega o modelo, e run_once roda a detecção sobre um frame e retorna a
# média das confidences das boxes detectadas. Chamada por candidates() para
# montar a lista de candidatos comparados por backend_benchmark.py.
def _fastsam_candidate(checkpoint: str, frames: list[Path]) -> Candidate:
    frame_iter = _frame_cycle(frames)

    # Carrega o modelo FastSAM do checkpoint indicado; chamada uma única vez
    # por benchmark_candidate.
    def factory() -> object:
        from ultralytics import FastSAM

        model = FastSAM(checkpoint)
        return model

    # Roda a detecção do FastSAM sobre o próximo frame do ciclo e retorna a
    # média das confidences das boxes detectadas (0.0 se nenhuma box for
    # encontrada); é o run_once exigido pelo contrato de benchmark_candidate.
    def run_once(model: object) -> float:
        image = next(frame_iter)
        results = model(np.array(image), device=0, retina_masks=True, verbose=False)  # type: ignore[operator]
        result = results[0]
        if result.boxes is None or result.boxes.conf is None or len(result.boxes.conf) == 0:
            return 0.0
        confidences = result.boxes.conf.detach().cpu().numpy()
        return float(confidences.mean())

    return (f"fastsam:{checkpoint}", factory, run_once)


# Monta a lista de candidatos de region discovery avaliados pelo benchmark da
# #174: duas variantes SAM/SAM2 e um FastSAM. Chamada por
# run_backend_benchmark.py ao selecionar o stage "region_discovery".
def candidates(frames: list[Path]) -> list[Candidate]:
    return [
        _sam_family_candidate("sam:facebook/sam-vit-huge", "facebook/sam-vit-huge", frames),
        _sam_family_candidate(
            "sam2:facebook/sam2.1-hiera-large", "facebook/sam2.1-hiera-large", frames
        ),
        _fastsam_candidate("FastSAM-x.pt", frames),
    ]
