"""Payload de pixel resolvido, usado internamente por stages, ports e fakes.

``ImageObservation`` (ver #153) referencia dados de pixel só através de uma
``ArtifactReference`` e permanece agnóstico de implementação. Quando um
stage precisa de fato rodar inferência, ele resolve essa referência para um
``ImagePayload`` (um array de pixel concreto). Isso mantém o contract
público livre de tipos de biblioteca de imaging, ao mesmo tempo em que dá
aos ports algo real para operar.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Representa um array de pixel RGB concreto mais a resolução que ele
# declara. Existe como a forma resolvida (não referenciada) de imagem que os
# ports realmente processam.
@dataclass(frozen=True, eq=False)
class ImagePayload:
    """Um array de pixel RGB concreto mais a resolução que ele declara."""

    pixels: np.ndarray
    width: int
    height: int

    # Valida que o array de pixels tem shape (H, W, 3) e que bate com
    # height/width declarados.
    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError(f"pixels must have shape (H, W, 3), got {self.pixels.shape}.")
        if self.pixels.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"pixels shape {self.pixels.shape[:2]} does not match "
                f"(height, width) = ({self.height}, {self.width})."
            )

    # Recorta um payload local, escopado a um tile, para a caixa de pixel
    # semi-aberta dada. Usada pelo tiling para extrair a região de pixels de
    # cada tile antes de rodar inferência nele.
    def crop(self, x_min: int, y_min: int, x_max: int, y_max: int) -> ImagePayload:
        """Retorna um payload local, escopado a um tile, para a caixa de pixel semi-aberta dada."""
        sub = self.pixels[y_min:y_max, x_min:x_max]
        return ImagePayload(sub, width=sub.shape[1], height=sub.shape[0])
