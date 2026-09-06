"""Condições de captura medidas a partir do próprio frame.

Issue: #197.

O manifest de referência exige registrar a condição de captura de cada
amostra. Um anotador humano poderia escrevê-la, mas duas propriedades — o
quanto o frame está escuro e o quanto está borrado — são medíveis
diretamente dos pixels, de forma determinística e reproduzível. Medi-las é
melhor do que pedi-las: não depende de julgamento, e o mesmo frame sempre
recebe a mesma condição.

Nada aqui infere semântica: estas são propriedades fotométricas do sinal, e
não rótulos sobre o conteúdo da cena.
"""

from __future__ import annotations

import numpy as np

#: Brilho médio (em ``[0, 1]``) abaixo do qual o frame é registrado como
#: pouco iluminado. Calibrado contra a vinheta do fisheye do corridor-02, que
#: já escurece as bordas de todo frame da sequência.
LOW_LIGHT_MEAN = 0.35

#: Variância do laplaciano abaixo da qual o frame é registrado como borrado.
#: Um valor pequeno significa poucas bordas nítidas, que é o efeito de
#: movimento rápido em uma câmera montada no robô.
BLUR_LAPLACIAN_VARIANCE = 60.0


# Mede as condições fotométricas de um frame e devolve os rótulos de
# condição aplicáveis. Chamada por prepare_reference ao construir cada
# amostra do manifest.
def measure_conditions(pixels: np.ndarray) -> tuple[str, ...]:
    """Retorna as condições de captura medidas neste frame.

    Argumentos:
        pixels: array RGB ``(H, W, 3)`` do frame.
    Retorna:
        tupla ordenada de condições, sempre contendo uma de iluminação e uma
        de nitidez, para que a ausência de rótulo nunca seja ambígua.
    """
    luminance = _luminance(pixels)
    conditions = ["low_light" if float(luminance.mean()) < LOW_LIGHT_MEAN else "normal_light"]
    conditions.append(
        "motion_blur" if laplacian_variance(luminance) < BLUR_LAPLACIAN_VARIANCE else "sharp"
    )
    return tuple(conditions)


# Calcula a variância do laplaciano de 4 vizinhos, a medida clássica de
# nitidez de imagem. Implementada com fatiamento numpy para que o pacote não
# precise de scipy nem de OpenCV.
def laplacian_variance(luminance: np.ndarray) -> float:
    """Retorna a variância do laplaciano da luminância, em escala 0-255."""
    scaled = luminance * 255.0
    laplacian = (
        scaled[:-2, 1:-1] + scaled[2:, 1:-1] + scaled[1:-1, :-2] + scaled[1:-1, 2:] - 4.0 * scaled[1:-1, 1:-1]
    )
    return float(laplacian.var()) if laplacian.size else 0.0


# Converte pixels RGB em luminância normalizada em [0, 1] usando os pesos
# padrão de percepção. Helper compartilhado pelas duas medidas.
def _luminance(pixels: np.ndarray) -> np.ndarray:
    """Retorna a luminância normalizada do frame."""
    values = pixels.astype(np.float64) / 255.0
    return values[:, :, 0] * 0.299 + values[:, :, 1] * 0.587 + values[:, :, 2] * 0.114
