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

from visual_perception.domain.arrays import read_only_view


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
    # height/width declarados, e guarda os pixels como visão somente-leitura:
    # todo recorte derivado herda a proteção, então nenhum adapter consegue
    # alterar a imagem de origem das views seguintes.
    def __post_init__(self) -> None:
        """Valida shape e resolução, e impede escrita nos pixels através do payload."""
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError(f"pixels must have shape (H, W, 3), got {self.pixels.shape}.")
        if self.pixels.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"pixels shape {self.pixels.shape[:2]} does not match "
                f"(height, width) = ({self.height}, {self.width})."
            )
        object.__setattr__(self, "pixels", read_only_view(self.pixels))

    # Compara dois payloads por conteúdo. Existe porque o tipo se apresenta
    # como value object; antes a igualdade era por identidade, e dois payloads
    # com os mesmos pixels eram diferentes.
    def __eq__(self, other: object) -> bool:
        """Indica se ``other`` tem a mesma resolução e os mesmos pixels."""
        if not isinstance(other, ImagePayload):
            return NotImplemented
        return (
            self.width == other.width
            and self.height == other.height
            and bool(np.array_equal(self.pixels, other.pixels))
        )

    #: Não hasheável pela mesma razão de ``Mask``: a igualdade é por conteúdo.
    __hash__ = None  # type: ignore[assignment]

    # Recorta um payload local, escopado a um tile, para a caixa de pixel
    # semi-aberta dada. Usada pelo tiling para extrair a região de pixels de
    # cada tile antes de rodar inferência nele.
    def crop(self, x_min: int, y_min: int, x_max: int, y_max: int) -> ImagePayload:
        """Retorna um payload local, escopado a um tile, para a caixa de pixel semi-aberta dada."""
        sub = self.pixels[y_min:y_max, x_min:x_max]
        return ImagePayload(sub, width=sub.shape[1], height=sub.shape[0])
