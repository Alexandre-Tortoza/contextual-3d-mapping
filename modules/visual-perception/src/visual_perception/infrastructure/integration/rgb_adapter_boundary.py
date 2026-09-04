"""Integra observações RGB canônicas vindas dos adapters do repositório.

Issue: #177.

A fronteira ``CanonicalObservation`` de `[adapters]` #103 já existe
(`contextual_mapping_adapters`). Este módulo adapta uma ``CanonicalObservation``
do tipo RGB (mais seu array de pixels já resolvido — decodificar a artifact
URI referenciada é específico de dataset/transporte e fica fora de
visual-perception, conforme #151) para a entrada canônica do próprio módulo.
"""

from __future__ import annotations

import numpy as np
from contextual_mapping_adapters import CanonicalObservation

from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload


# Converte uma CanonicalObservation RGB (do boundary de adapters do
# repositório) para o par (ImageObservation, ImagePayload) que o pipeline de
# visual-perception espera como entrada. É o ponto único onde o formato
# externo de adapters entra no módulo.
def to_canonical_input(
    observation: CanonicalObservation, pixels: np.ndarray, encoding: str = "rgb8"
) -> tuple[ImageObservation, ImagePayload]:
    """Adapta uma ``CanonicalObservation`` RGB já resolvida para a entrada de visual-perception.

    Falha antes da inferência quando metadados obrigatórios estão ausentes,
    conforme os critérios de aceitação de #177, delegando para a própria
    validação de :class:`ImageObservation`.
    """
    if observation.kind != "rgb":
        raise ValueError(f"Expected an rgb CanonicalObservation, got kind={observation.kind!r}.")
    height, width = pixels.shape[:2]
    image_observation = ImageObservation(
        width=width,
        height=height,
        encoding=encoding,
        image=observation.artifact,
        source=observation.reference,
    )
    payload = ImagePayload(pixels, width=width, height=height)
    return image_observation, payload
