"""Posse de arrays numpy dentro dos value objects de domínio."""

from __future__ import annotations

import numpy as np


# Devolve uma visão somente-leitura do array, sem copiar. Existe porque os
# value objects de domínio são ``frozen``, mas ``frozen`` só impede reatribuir
# o campo: sem isto, ``mask.data[0, 0] = False`` alterava uma máscara
# congelada, e um adapter que escrevesse in-place corrompia a imagem de origem
# de todas as views seguintes. A visão não altera o array de quem construiu o
# objeto; ela impede a escrita *através* do objeto. Usada por Mask,
# ImagePayload e FeatureMap.
def read_only_view(array: np.ndarray) -> np.ndarray:
    """Retorna uma visão do array que recusa escrita.

    Argumentos:
        array: o array entregue ao value object.
    Retorna:
        uma visão sobre os mesmos dados, com ``writeable`` desligado.
    """
    view = array.view()
    view.flags.writeable = False
    return view
