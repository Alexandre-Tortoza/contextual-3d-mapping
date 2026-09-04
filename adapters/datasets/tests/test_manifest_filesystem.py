import json
from pathlib import Path

import pytest

from contextual_mapping_adapters import DatasetFilesystem
from contextual_mapping_datasets import load_dataset_manifest


# Grava um manifest mínimo que representa o formato rastreado no repositório.
# Existe para os testes exercitarem o loader sem depender de um dataset externo.
def _write_manifest(path: Path, artifact_uri: str = "sequence/camera.png") -> None:
    """Grava um manifest mínimo para os cenários de teste."""
    path.write_text(
        json.dumps(
            {
                "dataset_id": "fixture-dataset",
                "sequences": [
                    {
                        "sequence_id": "sequence-01",
                        "calibrations": [
                            {
                                "calibration_id": "camera-calibration",
                                "artifact_uri": "calibration/camera.json",
                                "source_frame": "camera",
                                "target_frame": "body",
                            }
                        ],
                        "sensors": [
                            {
                                "sensor_id": "camera",
                                "kind": "rgb",
                                "artifact_uri": artifact_uri,
                                "media_type": "image/png",
                                "frame_id": "camera",
                                "clock_id": "clock",
                                "calibration_id": "camera-calibration",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


# Confere que o loader preserva referências relativas no manifest, para que a
# máquina que executa o adapter seja quem determine a raiz local do dataset.
def test_load_dataset_manifest_builds_a_typed_manifest(tmp_path: Path) -> None:
    """Verifica o carregamento de um manifest JSON válido."""
    manifest_path = tmp_path / "fixture-dataset.json"
    _write_manifest(manifest_path)

    manifest = load_dataset_manifest(manifest_path)

    assert manifest.dataset_id == "fixture-dataset"
    assert manifest.sequence("sequence-01").sensors[0].artifact_uri == "sequence/camera.png"


# Confere que o filesystem resolve um artifact apenas sob a raiz canônica e
# emite a referência pública `file:` que uma observação canônica transporta.
def test_dataset_filesystem_resolves_existing_manifest_artifact(tmp_path: Path) -> None:
    """Verifica a resolução confinada de um artifact existente."""
    manifest_path = tmp_path / "fixture-dataset.json"
    _write_manifest(manifest_path)
    artifact = tmp_path / "datasets/raw/fixture-dataset/sequence/camera.png"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"image")
    manifest = load_dataset_manifest(manifest_path)

    filesystem = DatasetFilesystem(tmp_path, manifest)
    reference = filesystem.source_artifact("sequence/camera.png", "image/png")

    assert filesystem.raw_root == tmp_path / "datasets/raw/fixture-dataset"
    assert reference.uri == artifact.as_uri()
    assert reference.media_type == "image/png"


# Confere que o loader rejeita uma referência que sairia da raiz antes que um
# adapter de formato possa processar o manifest ou abrir um arquivo local.
def test_load_dataset_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    """Verifica a rejeição de traversal no artifact do manifest."""
    manifest_path = tmp_path / "unsafe.json"
    _write_manifest(manifest_path, "../outside.png")

    with pytest.raises(ValueError, match="path traversal"):
        load_dataset_manifest(manifest_path)


# Confere que a camada de filesystem também não aceita traversal, protegendo
# adapters que tenham construído um manifest por código em vez do loader JSON.
def test_dataset_filesystem_rejects_path_traversal_from_direct_call(tmp_path: Path) -> None:
    """Verifica a proteção do resolver contra traversal direto."""
    manifest_path = tmp_path / "fixture-dataset.json"
    _write_manifest(manifest_path)
    raw_root = tmp_path / "datasets/raw/fixture-dataset"
    raw_root.mkdir(parents=True)
    filesystem = DatasetFilesystem(tmp_path, load_dataset_manifest(manifest_path))

    with pytest.raises(ValueError, match="relative local path"):
        filesystem.resolve("sequence/../camera.png")
