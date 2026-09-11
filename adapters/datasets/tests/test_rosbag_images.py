"""Testes dos contracts leves de inspeção de imagens em rosbags."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contextual_mapping_adapters import (
    RosbagImageTopic,
    RosbagRecording,
    decode_image_message,
    select_rgb_topic,
)


# A detecção deve preferir o stream RGB principal e nunca escolher profundidade
# apenas porque ela aparece primeiro na ordenação do arquivo.
def test_select_rgb_topic_prefere_stream_principal() -> None:
    """Confere seleção automática e override explícito do tópico RGB."""
    recording = RosbagRecording(
        Path("run.bag"),
        10.0,
        (
            RosbagImageTopic("/camera/depth", "sensor_msgs/msg/Image", 1000),
            RosbagImageTopic("/camera/rgb", "sensor_msgs/msg/Image", 900),
            RosbagImageTopic("/camera/preview", "sensor_msgs/msg/Image", 10),
        ),
    )
    assert select_rgb_topic(recording).name == "/camera/rgb"
    assert select_rgb_topic(recording, "/camera/preview").message_count == 10


# Um tópico desconhecido precisa falhar antes da extração percorrer uma rosbag
# potencialmente grande.
def test_select_rgb_topic_rejeita_override_desconhecido() -> None:
    """Confere diagnóstico antecipado para tópico inválido."""
    recording = RosbagRecording(
        Path("run.bag"),
        1.0,
        (RosbagImageTopic("/rgb", "sensor_msgs/msg/Image", 1),),
    )
    with pytest.raises(ValueError, match="não é um stream"):
        select_rgb_topic(recording, "/missing")


# O reorder BGR é uma invariante observada no corridor-02 e deve permanecer no
# adapter quando o harness deixa de possuir sua própria implementação.
def test_decode_image_message_converte_bgr_para_rgb() -> None:
    """Confere a ordem dos canais de uma mensagem bgr8."""
    message = SimpleNamespace(height=1, width=1, encoding="bgr8", data=bytes((1, 2, 3)))
    pixels = decode_image_message(message, "sensor_msgs/msg/Image")
    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [[[3, 2, 1]]]
