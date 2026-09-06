"""Leitura de manifests JSON versionados de datasets."""

from __future__ import annotations

import json
from pathlib import Path

from ._document import (
    mapping as _mapping,
)
from ._document import (
    optional_string as _optional_string,
)
from ._document import (
    relative_artifact_uri as _relative_artifact_uri,
)
from ._document import (
    sequence as _sequence,
)
from ._document import (
    string as _string,
)
from .manifest import (
    SUPPORTED_SCHEMA_VERSION,
    CalibrationManifest,
    DatasetManifest,
    SensorSourceManifest,
    SequenceManifest,
)


# Carrega e valida um manifest JSON rastreado no repositório. Existe para
# separar a desserialização de arquivos da representação tipada consumida por
# adapters e experimentos, garantindo uma falha acionável na fronteira.
def load_dataset_manifest(path: Path) -> DatasetManifest:
    """Carrega um `DatasetManifest` a partir de um arquivo JSON.

    Argumentos:
        path: arquivo JSON versionado sob `datasets/manifests`.
    Retorna:
        manifest tipado e validado para uso por adapters e experimentos.
    Levanta:
        FileNotFoundError: se o arquivo de manifest não existir.
        ValueError: se o JSON ou sua estrutura não forem válidos.
    """
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest {path}: {error.msg}.") from error
    return dataset_manifest_from_mapping(document, source=str(path))


# Converte o documento já desserializado na representação tipada do domínio.
# É pública para que ferramentas de experimento possam validar manifests antes
# de gravá-los, sem duplicar as regras de parsing do loader de arquivo.
def dataset_manifest_from_mapping(document: object, *, source: str = "manifest") -> DatasetManifest:
    """Converte um mapeamento JSON em um manifest validado.

    Argumentos:
        document: objeto desserializado de um documento JSON.
        source: identificação usada nas mensagens de validação.
    Retorna:
        manifest tipado e validado.
    Levanta:
        ValueError: se campos obrigatórios ou referências locais forem inválidos.
    """
    root = _mapping(document, source)
    dataset_id = _string(root, "dataset_id", source)
    schema_version = _string(root, "schema_version", source, default=SUPPORTED_SCHEMA_VERSION)
    source_uri = _optional_string(root, "source_uri", source)
    sequences_value = _sequence(root, "sequences", source)
    sequences = tuple(
        _sequence_manifest(item, f"{source}.sequences[{index}]")
        for index, item in enumerate(sequences_value)
    )
    return DatasetManifest(dataset_id, sequences, schema_version, source_uri)


# Constrói uma sequência e seus contracts locais a partir de um item do
# documento. A validação de ids e calibrações permanece nos dataclasses do
# domínio, que são a fonte de verdade dessas invariantes.
def _sequence_manifest(value: object, source: str) -> SequenceManifest:
    document = _mapping(value, source)
    calibrations = tuple(
        _calibration_manifest(item, f"{source}.calibrations[{index}]")
        for index, item in enumerate(_sequence(document, "calibrations", source, default=()))
    )
    sensors = tuple(
        _sensor_source_manifest(item, f"{source}.sensors[{index}]")
        for index, item in enumerate(_sequence(document, "sensors", source))
    )
    return SequenceManifest(
        _string(document, "sequence_id", source),
        sensors,
        calibrations,
        _optional_string(document, "split", source),
    )


# Constrói a referência de calibração e restringe seu artifact a um caminho
# relativo. Calibrações baixadas fazem parte do dataset bruto e não podem
# apontar para arquivos arbitrários da máquina que executa o adapter.
def _calibration_manifest(value: object, source: str) -> CalibrationManifest:
    document = _mapping(value, source)
    artifact_uri = _relative_artifact_uri(_string(document, "artifact_uri", source), source)
    return CalibrationManifest(
        _string(document, "calibration_id", source),
        artifact_uri,
        _string(document, "source_frame", source),
        _string(document, "target_frame", source),
    )


# Constrói uma fonte de sensor declarada no manifest. O caminho continua como
# uma URI relativa no contract e só é resolvido para filesystem pelo adapter
# que recebeu explicitamente a raiz local do repositório.
def _sensor_source_manifest(value: object, source: str) -> SensorSourceManifest:
    document = _mapping(value, source)
    artifact_uri = _relative_artifact_uri(_string(document, "artifact_uri", source), source)
    calibration_id = _optional_string(document, "calibration_id", source)
    required = document.get("required", True)
    if not isinstance(required, bool):
        raise ValueError(f"{source}.required must be a boolean.")
    return SensorSourceManifest(
        _string(document, "sensor_id", source),
        _string(document, "kind", source),
        artifact_uri,
        _string(document, "media_type", source),
        _string(document, "frame_id", source),
        _string(document, "clock_id", source),
        calibration_id,
        required,
    )
