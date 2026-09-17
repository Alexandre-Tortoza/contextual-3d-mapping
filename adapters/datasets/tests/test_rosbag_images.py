"""Testes dos contracts leves de inspeção de imagens em rosbags."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from contextual_mapping_adapters import (
    ExtractedFrameProvenance,
    RosbagImageTopic,
    RosbagRecording,
    decode_image_message,
    extract_strided_rosbag_frames,
    frame_provenance_path,
    read_frame_provenance,
    select_rgb_topic,
    write_frame_provenance,
)
from contextual_mapping_adapters.rosbag_images import _header_frame_id


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


# O registro de proveniência é o que permite a um consumidor montar a
# observação com o timestamp real do frame; ida e volta não pode perder nada.
def test_frame_provenance_round_trip(tmp_path: Path) -> None:
    """Confere que o registro escrito na extração é lido de volta igual."""
    image_path = tmp_path / "corridor-02-008.png"
    image_path.touch()
    provenance = ExtractedFrameProvenance(
        recording_id="corridor-02",
        topic="/camera_1/image_raw",
        frame_id="camera_1_optical_frame",
        timestamp_ns=1_724_621_103_744_773_747,
        sequence_index=355,
    )

    written = write_frame_provenance(image_path, provenance)

    assert written == frame_provenance_path(image_path) == tmp_path / "corridor-02-008.json"
    assert read_frame_provenance(image_path) == provenance


# Frames extraídos antes do registro existir não têm de onde tirar timestamp;
# a leitura precisa dizer isso, e não devolver um valor inventado.
def test_frame_provenance_missing_record_is_actionable(tmp_path: Path) -> None:
    """Confere que a ausência do registro falha pedindo nova extração."""
    with pytest.raises(FileNotFoundError, match="Re-extract"):
        read_frame_provenance(tmp_path / "frame.png")


# Um registro de outro formato ou com valores fora do domínio falha na
# leitura, antes de chegar a um contract de observação.
@pytest.mark.parametrize(
    "record",
    [
        {"schema": "other/1"},
        {"schema": "rosbag-frame-provenance/1", "recording_id": "bag"},
        {
            "schema": "rosbag-frame-provenance/1",
            "recording_id": "bag",
            "topic": "/rgb",
            "frame_id": None,
            "timestamp_ns": -1,
            "sequence_index": 0,
        },
    ],
)
def test_frame_provenance_rejects_invalid_records(tmp_path: Path, record: dict[str, object]) -> None:
    """Confere a rejeição de formato desconhecido, campo ausente e valor inválido."""
    (tmp_path / "frame.json").write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError):
        read_frame_provenance(tmp_path / "frame.png")


# O frame de coordenadas vem do header da mensagem; header vazio vira ``None``
# para que o consumidor decida, em vez de receber uma string vazia.
def test_header_frame_id_is_read_from_the_message() -> None:
    """Confere a leitura de ``header.frame_id`` com e sem valor."""
    assert _header_frame_id(SimpleNamespace(header=SimpleNamespace(frame_id="camera"))) == "camera"
    assert _header_frame_id(SimpleNamespace(header=SimpleNamespace(frame_id=""))) is None
    assert _header_frame_id(SimpleNamespace()) is None


# Reader fake de um stream com 10 mensagens de 1 pixel, suficiente para
# exercitar seleção de faixa e passo sem abrir uma rosbag real.
class _FakeAnyReader:
    def __init__(self, paths: object) -> None:
        self.connections = [SimpleNamespace(topic="/camera/rgb", msgtype="sensor_msgs/msg/Image")]
        self.topics = {"/camera/rgb": SimpleNamespace(msgcount=10)}
        self.start_time = 0
        self.end_time = 10_000_000_000

    def __enter__(self) -> "_FakeAnyReader":
        return self

    def __exit__(self, *exception_info: object) -> bool:
        return False

    def messages(self, connections: list[object]) -> list[tuple[object, int, bytes]]:
        return [(connections[0], index * 1_000_000, b"raw") for index in range(10)]

    def deserialize(self, rawdata: bytes, msgtype: str) -> SimpleNamespace:
        return SimpleNamespace(
            height=1, width=1, encoding="rgb8", data=bytes((1, 2, 3)), header=SimpleNamespace(frame_id=""),
        )


# O nome do arquivo precisa carregar o índice real no stream, não a ordem de
# escrita: é isso que permite a duas extrações em lotes diferentes do mesmo
# bag produzirem os mesmos nomes para os mesmos frames (pré-condição do
# resume no harness de percepção).
def test_extract_strided_rosbag_frames_names_by_stream_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Confere passo, faixa e nomeação por índice real do stream."""
    monkeypatch.setattr("rosbags.highlevel.AnyReader", _FakeAnyReader)
    bag_path = tmp_path / "corridor-02.bag"
    bag_path.touch()
    output_dir = tmp_path / "frames"
    written = extract_strided_rosbag_frames(
        bag_path, output_dir, stride=3, start_index=2, end_index=9, frame_id_prefix="corridor-02",
    )
    # O passo é absoluto (índice % stride == 0), não relativo a start_index:
    # é isso que garante que lotes diferentes (faixas diferentes) do mesmo
    # bag selecionem exatamente os mesmos índices globais.
    assert [path.name for path in written] == ["corridor-02-00003.png", "corridor-02-00006.png"]
    for path in written:
        assert path.is_file()
        assert read_frame_provenance(path).sequence_index == int(path.stem.rsplit("-", 1)[1])


def test_extract_strided_rosbag_frames_rejects_invalid_stride_or_range(tmp_path: Path) -> None:
    """Confere a rejeição de ``stride`` não positivo e de faixa invertida."""
    with pytest.raises(ValueError, match="stride"):
        extract_strided_rosbag_frames(Path("run.bag"), tmp_path, stride=0)
    with pytest.raises(ValueError, match="faixa de índices"):
        extract_strided_rosbag_frames(Path("run.bag"), tmp_path, stride=1, start_index=5, end_index=5)
