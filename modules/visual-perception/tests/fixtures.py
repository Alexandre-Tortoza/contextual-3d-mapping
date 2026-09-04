"""Fixtures compartilhadas para observações canônicas de imagem e visuais (#173)."""

from __future__ import annotations

import numpy as np
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp

from visual_perception.config import ModuleConfig
from visual_perception.domain.image_observation import ImageObservation
from visual_perception.domain.image_payload import ImagePayload


# Gera um payload totalmente branco; existe porque o fake region discoverer não
# encontra nenhuma região nele, servindo como caso "imagem vazia" para os testes.
def blank_payload(width: int = 32, height: int = 32) -> ImagePayload:
    """Uma imagem toda branca: o fake region discoverer encontra zero regiões nela."""
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    return ImagePayload(pixels, width=width, height=height)


# Gera um payload branco com um ou mais blobs retangulares de cor sólida; usada pelos
# testes que precisam de regiões detectáveis (contraste contra o fundo branco) sem
# depender de uma imagem real.
def payload_with_blobs(
    width: int = 32, height: int = 32, blobs: tuple[tuple[int, int, int, int, tuple[int, int, int]], ...] = (
        (4, 4, 10, 10, (200, 30, 30)),
    ),
) -> ImagePayload:
    """Uma imagem branca com um ou mais blobs retangulares de cor sólida.

    Cada blob é ``(x_min, y_min, x_max, y_max, rgb)``.
    """
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    for x_min, y_min, x_max, y_max, rgb in blobs:
        pixels[y_min:y_max, x_min:x_max] = rgb
    return ImagePayload(pixels, width=width, height=height)


# Constrói uma ImageObservation canônica mínima e válida, com as referências de fonte
# (source/observation) já preenchidas; usada como ponto de partida por praticamente
# todos os testes que precisam de uma observação de imagem.
def image_observation(
    observation_id: str = "frame-0001", width: int = 32, height: int = 32
) -> ImageObservation:
    image = SourceArtifactReference(uri="mem://frame-0001", media_type="image/png")
    source = ObservationReference(
        observation_id=observation_id,
        dataset_id="corridor02",
        sequence_id="seq-0",
        sensor_id="camera_1",
        sequence_index=0,
        timestamp=Timestamp(nanoseconds=1_000_000, clock_id="rosbag"),
        frame_id=FrameId("camera_1_optical_frame"),
    )
    return ImageObservation(width=width, height=height, encoding="rgb8", image=image, source=source)


# Retorna uma ModuleConfig com todos os defaults; existe para dar aos testes uma
# configuração válida e neutra sem precisar repetir a construção em cada um.
def default_config() -> ModuleConfig:
    return ModuleConfig()
