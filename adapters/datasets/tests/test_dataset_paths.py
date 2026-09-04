from pathlib import Path

import pytest

from contextual_mapping_datasets import (
    RAW_DATASETS_DIRECTORY,
    CalibrationManifest,
    DatasetManifest,
    SensorSourceManifest,
    SequenceManifest,
    raw_dataset_root,
)


# Confere que todos os consumidores derivam a mesma raiz local de dataset,
# evitando que adapters e experimentos adotem layouts incompatíveis.
def test_raw_dataset_root_uses_the_repository_convention() -> None:
    """Verifica a resolução da raiz canônica de um dataset."""
    repository_root = Path("/workspace/contextual-3d-mapping")

    assert RAW_DATASETS_DIRECTORY == Path("datasets/raw")
    assert raw_dataset_root(repository_root, "cerberus-subt") == (
        repository_root / "datasets/raw/cerberus-subt"
    )


# Confere que o identificador não pode escapar da raiz canônica por meio de
# segmentos de caminho, antes que um adapter tente resolver artifacts locais.
@pytest.mark.parametrize("dataset_name", ("", ".", "..", "../outside", "/tmp/outside", "one/two"))
def test_raw_dataset_root_rejects_non_segment_dataset_names(dataset_name: str) -> None:
    """Verifica a rejeição de nomes que não são segmentos de caminho."""
    with pytest.raises(ValueError):
        raw_dataset_root(Path("/workspace/contextual-3d-mapping"), dataset_name)


# Confere que o id presente no manifest obedece à mesma regra de localização
# do diretório bruto, evitando que a validação seja esquecida por um adapter.
def test_manifest_rejects_dataset_id_that_is_not_a_path_segment() -> None:
    """Verifica que o id do manifest segue a convenção da raiz local."""
    sequence = SequenceManifest(
        "sequence",
        (
            SensorSourceManifest(
                "camera", "rgb", "file:///camera", "image/png", "camera", "clock", "camera-calibration"
            ),
        ),
        (CalibrationManifest("camera-calibration", "file:///calibration", "camera", "body"),),
    )

    with pytest.raises(ValueError, match="single relative path segment"):
        DatasetManifest("../outside", (sequence,))
