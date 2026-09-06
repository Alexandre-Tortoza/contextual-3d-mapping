"""Decodificação e geometria de máscara para a avaliação de percepção visual.

Issue: #198.

A avaliação lê máscaras de duas origens — o manifest de referência (#197) e
o contract de predição (``prediction.py``) — e as duas as codificam em RLE
começando por ``False``. Este módulo é o único lugar que decodifica esse
formato e o único que sabe extrair contorno, para que uma correção na
geometria valha para os dois lados ao mesmo tempo.

NOTE: o decodificador é deliberadamente independente do de
``visual_perception.infrastructure.serialization``. A #198 exige que as
métricas consumam *apenas* contracts versionados; depender do codec interno
do módulo avaliado faria a avaliação seguir mudanças internas dele. A
compatibilidade entre os dois lados é fixada por um teste de round-trip no
próprio módulo.
"""

from __future__ import annotations

import numpy as np


# Reconstrói a máscara booleana a partir da codificação RLE compartilhada
# pelo manifest de referência e pelo contract de predição. Existe como a
# única implementação do decodificador usada pela avaliação.
def decode_mask(width: int, height: int, runs: tuple[int, ...]) -> np.ndarray:
    """Decodifica um RLE que começa em ``False`` em um array booleano ``(height, width)``.

    Argumentos:
        width: largura da imagem em pixels.
        height: altura da imagem em pixels.
        runs: comprimentos alternados começando pelo primeiro trecho ``False``.
    Retorna:
        array booleano com shape ``(height, width)``.
    Levanta:
        ValueError: se as dimensões forem inválidas ou o RLE não cobrir a imagem.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Mask dimensions must be positive.")
    if not runs or any(int(run) < 0 for run in runs):
        raise ValueError("Mask runs must be non-negative integers.")
    total = width * height
    if sum(int(run) for run in runs) != total:
        raise ValueError(f"Mask runs must cover exactly {total} pixels, got {sum(runs)}.")
    flat = np.zeros(total, dtype=np.bool_)
    cursor, value = 0, False
    for run in runs:
        if value and run:
            flat[cursor : cursor + int(run)] = True
        cursor += int(run)
        value = not value
    return flat.reshape(height, width)


# Calcula a intersecção-sobre-união entre duas máscaras da mesma resolução.
# Existe como o núcleo da associação predição/referência usada por todas as
# métricas de região.
def intersection_over_union(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Retorna a IoU entre duas máscaras booleanas de mesma resolução."""
    if reference.shape != prediction.shape:
        raise ValueError("Masks must share the same resolution to be compared.")
    union = int(np.logical_or(reference, prediction).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(reference, prediction).sum()) / float(union)


# Extrai o contorno de uma máscara: os pixels ocupados que tocam o fundo em
# alguma das quatro direções. Existe para a métrica de boundary F1, que mede
# qualidade de borda separada de qualidade de área.
def boundary(mask: np.ndarray) -> np.ndarray:
    """Retorna os pixels de ``mask`` que fazem fronteira com o fundo (vizinhança-4)."""
    if not mask.any():
        return np.zeros_like(mask)
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    return mask & ~interior


# Dilata uma máscara por uma tolerância em pixels na métrica de Chebyshev
# (um quadrado de lado ``2 * tolerance + 1``). Existe para que a comparação
# de contorno admita um deslocamento pequeno sem exigir coincidência exata,
# que nenhuma anotação humana satisfaz.
def dilate(mask: np.ndarray, tolerance: int) -> np.ndarray:
    """Dilata ``mask`` por ``tolerance`` pixels em todas as direções.

    A dilatação é feita sobre uma cópia com padding de fundo, e não por
    rolagem: rolar faria a borda de cima reaparecer na de baixo e inventar
    contorno onde a imagem termina.
    """
    if tolerance <= 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, tolerance, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask)
    for offset_y in range(2 * tolerance + 1):
        for offset_x in range(2 * tolerance + 1):
            dilated |= padded[offset_y : offset_y + height, offset_x : offset_x + width]
    return dilated


# Calcula o boundary F1 entre duas máscaras: quanto do contorno previsto cai
# perto do contorno de referência, e vice-versa. Existe porque IoU de área
# quase não penaliza uma borda ruim em uma região grande, que é exatamente o
# que a evidência de alta resolução (#192) deveria melhorar.
def boundary_f1(reference: np.ndarray, prediction: np.ndarray, *, tolerance: int = 2) -> float:
    """Retorna o F1 entre os contornos de ``reference`` e ``prediction``.

    Argumentos:
        reference: máscara anotada.
        prediction: máscara prevista, na mesma resolução.
        tolerance: deslocamento em pixels admitido entre os contornos.
    Retorna:
        o F1 de contorno em ``[0, 1]``; ``0.0`` quando só um dos lados tem contorno,
        e ``1.0`` quando os dois estão vazios (nada a errar).
    """
    if reference.shape != prediction.shape:
        raise ValueError("Masks must share the same resolution to be compared.")
    reference_boundary = boundary(reference)
    prediction_boundary = boundary(prediction)
    reference_total = int(reference_boundary.sum())
    prediction_total = int(prediction_boundary.sum())
    if reference_total == 0 and prediction_total == 0:
        return 1.0
    if reference_total == 0 or prediction_total == 0:
        return 0.0
    matched_prediction = int((prediction_boundary & dilate(reference_boundary, tolerance)).sum())
    matched_reference = int((reference_boundary & dilate(prediction_boundary, tolerance)).sum())
    precision = matched_prediction / prediction_total
    recall = matched_reference / reference_total
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
