"""Convenções de localização local para dados brutos de datasets."""

from __future__ import annotations

from pathlib import Path


# Representa o diretório relativo, versionado como convenção do repositório,
# onde ficam os dados brutos locais. Existe para que adapters, experimentos e
# manifests não reconstruam nem divirjam do layout `datasets/raw/<nome>/`.
RAW_DATASETS_DIRECTORY = Path("datasets/raw")


# Valida o nome que identifica um dataset no diretório `raw`. Existe para
# impedir que um id de dataset altere a raiz prevista por meio de segmentos de
# caminho, mantendo a resolução de arquivos locais confinada ao dataset dono.
def validate_dataset_name(dataset_name: str) -> None:
    """Valida um nome de dataset que será usado como segmento de caminho.

    Argumentos:
        dataset_name: identificador estável do dataset no layout local.
    Levanta:
        TypeError: se o nome não for uma string.
        ValueError: se o nome estiver vazio ou não representar um único segmento.
    """
    if not isinstance(dataset_name, str):
        raise TypeError("dataset_name must be a string.")
    if not dataset_name.strip():
        raise ValueError("dataset_name must not be empty.")
    candidate = Path(dataset_name)
    if candidate.is_absolute() or len(candidate.parts) != 1 or dataset_name in {".", ".."}:
        raise ValueError("dataset_name must be a single relative path segment.")


# Resolve a raiz local canônica de um dataset a partir da raiz do repositório.
# É usada por adapters de dataset e experimentos antes de abrirem artifacts,
# para que a localização física de dados brutos seja uniforme e explícita.
def raw_dataset_root(repository_root: Path, dataset_name: str) -> Path:
    """Retorna `datasets/raw/<dataset_name>` sob a raiz informada.

    Argumentos:
        repository_root: diretório raiz do repositório de trabalho.
        dataset_name: identificador estável do dataset.
    Retorna:
        caminho absoluto ou relativo à raiz local canônica do dataset.
    Levanta:
        ValueError: se o nome não puder compor um único segmento de caminho.
    """
    validate_dataset_name(dataset_name)
    return repository_root / RAW_DATASETS_DIRECTORY / dataset_name
