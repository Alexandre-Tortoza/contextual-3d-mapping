"""Candidatos de embedding alinhado à linguagem para o benchmark da #174.

Proxy de qualidade: margem de similaridade de cosseno zero-shot entre top1/top2
contra um vocabulário fixo e genérico de indoor/corridor. Uma distribuição de
similaridade mais "afiada" (peakier) significa que o embedding de fato
discrimina entre conteúdos de cena plausíveis, que é exatamente para o que
esse port serve (alinhamento região-para-texto) — e isso não precisa de
labels ground-truth de região, que este dataset não tem.

Cada frame funciona aqui como um substituto de "um crop de região": o pipeline
real chama ``encode_image`` em regiões recortadas por box (veja
``pipeline.py``), mas no momento de seleção de backend ainda não há proposals
de região reais, então o frame representativo inteiro é usado como entrada
substituta do mesmo tipo (uma imagem RGB que precisa de um embedding).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

Candidate = tuple[str, Callable[[], object], Callable[[object], float]]

_VOCABULARY = [
    "a wall",
    "a floor",
    "a ceiling",
    "a door",
    "a window",
    "a ceiling light fixture",
    "a fire extinguisher",
    "a pipe",
    "a mobile robot",
    "a person",
    "a plant",
    "a sign on the wall",
    "debris on the ground",
    "an exit sign",
    "a trash can",
    "a hallway corridor",
]


# Gera um iterador infinito que percorre ciclicamente os frames de referência,
# decodificando cada imagem como RGB sob demanda. Usada pelos candidatos deste
# módulo para alimentar run_once com uma nova imagem a cada chamada.
def _frame_cycle(frames: Iterable[Path]) -> Iterable[Image.Image]:
    for path in itertools.cycle(list(frames)):
        yield Image.open(path).convert("RGB")


# Normaliza a saída de get_text_features/get_image_features entre versões do
# transformers: extrai o tensor de embedding independente de vir embrulhado em
# um objeto com .pooler_output ou como tensor puro. Usada por run_once para
# obter os vetores de texto/imagem antes de calcular a similaridade de
# cosseno.
def _extract_features(output: object) -> torch.Tensor:
    """Desembrulha um resultado possivelmente pooled de ``get_*_features`` em um tensor simples.

    Versões mais novas de ``transformers`` (5.x) fazem ``get_text_features``/
    ``get_image_features`` retornarem um objeto no formato
    ``BaseModelOutputWithPooling``, cujo ``.pooler_output`` guarda o embedding
    projetado, em vez de um tensor puro; versões mais antigas retornavam o
    tensor diretamente.
    """
    pooler_output = getattr(output, "pooler_output", None)
    return pooler_output if pooler_output is not None else output  # type: ignore[return-value]


# Calcula a margem entre a maior e a segunda maior similaridade de cosseno
# (top1 - top2), limitada a [0, 1]. É o score de qualidade retornado por
# run_once: uma margem maior indica que o embedding discrimina melhor entre
# as classes do vocabulário candidato.
def _margin_from_similarities(similarities: np.ndarray) -> float:
    top_two = np.sort(similarities)[-2:]
    margin = float(top_two[1] - top_two[0])
    return max(0.0, min(1.0, margin))


# Constrói um candidato de benchmark (nome, factory, run_once) para um
# checkpoint da família CLIP/SigLIP: factory carrega o processor/modelo e
# pré-calcula os embeddings de texto normalizados do vocabulário fixo; run_once
# encoda cada frame e retorna a margem top1/top2 de similaridade (via
# _margin_from_similarities). Chamada por candidates() para montar a lista de
# candidatos comparados por backend_benchmark.py.
def _clip_family_candidate(name: str, checkpoint: str, frames: list[Path]) -> Candidate:
    frame_iter = _frame_cycle(frames)

    # Carrega o processor e o modelo do checkpoint indicado e pré-computa os
    # embeddings de texto normalizados do vocabulário fixo, para reuso em
    # todas as chamadas de run_once; chamada uma única vez por
    # benchmark_candidate.
    def factory() -> object:
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).to("cuda").eval()
        with torch.inference_mode():
            text_inputs = processor(text=_VOCABULARY, return_tensors="pt", padding=True).to("cuda")
            text_features = _extract_features(model.get_text_features(**text_inputs))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return (processor, model, text_features)

    # Encoda o próximo frame do ciclo, normaliza o embedding de imagem e
    # calcula sua similaridade de cosseno contra o vocabulário pré-computado;
    # é o run_once exigido pelo contrato de benchmark_candidate.
    def run_once(bundle: object) -> float:
        processor, model, text_features = bundle  # type: ignore[misc]
        image = next(frame_iter)
        image_inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            image_features = _extract_features(model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarities = (image_features @ text_features.T)[0].float().cpu().numpy()
        return _margin_from_similarities(similarities)

    return (name, factory, run_once)


# Monta a lista de candidatos de language-aligned embedding avaliados pelo
# benchmark da #174: um checkpoint CLIP e um SigLIP. Chamada por
# run_backend_benchmark.py ao selecionar o stage "language_embedding".
def candidates(frames: list[Path]) -> list[Candidate]:
    return [
        _clip_family_candidate(
            "clip:openai/clip-vit-large-patch14", "openai/clip-vit-large-patch14", frames
        ),
        _clip_family_candidate(
            "siglip:google/siglip-base-patch16-224", "google/siglip-base-patch16-224", frames
        ),
    ]
