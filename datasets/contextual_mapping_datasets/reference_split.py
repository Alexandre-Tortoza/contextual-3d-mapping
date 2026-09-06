"""Leitura e validação da partição versionada do conjunto de referência (#210)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from ._document import mapping, sequence
from ._document import string as json_string
from .annotation_manifest import ReferenceManifest, Split
from .annotation_manifest_io import load_reference_manifest

REFERENCE_SPLIT_SCHEMA_VERSION = "visual-reference-split/1"


# Representa a partição imutável separada do conteúdo anotado. Existe para
# detectar drift do manifest e vazamento antes de calibração ou avaliação.
@dataclass(frozen=True)
class ReferenceSplit:
    """IDs de development, calibration e test ligados ao digest do manifest."""

    reference_id: str
    source_manifest: str
    source_manifest_sha256: str
    partition_rule: str
    splits: dict[Split, tuple[str, ...]]
    schema_version: str = REFERENCE_SPLIT_SCHEMA_VERSION

    # Valida campos locais e garante que cada split exista e não repita IDs.
    def __post_init__(self) -> None:
        """Rejeita versão, digest, campos ou partições estruturalmente inválidos."""
        if self.schema_version != REFERENCE_SPLIT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported split schema_version {self.schema_version!r}; "
                f"expected {REFERENCE_SPLIT_SCHEMA_VERSION!r}."
            )
        for name in ("reference_id", "source_manifest", "partition_rule"):
            if not getattr(self, name).strip():
                raise ValueError(f"ReferenceSplit.{name} must not be empty.")
        digest = self.source_manifest_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("ReferenceSplit.source_manifest_sha256 must be a SHA-256 digest.")
        for split in Split:
            identifiers = self.splits.get(split, ())
            if not identifiers:
                raise ValueError(f"ReferenceSplit has no samples for {split.value!r}.")
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"ReferenceSplit repeats sample IDs in {split.value!r}.")


# Carrega a partição JSON em tipos fechados para que nomes de split inválidos
# falhem na fronteira, antes de qualquer experimento.
def load_reference_split(path: Path) -> ReferenceSplit:
    """Carrega e valida um artifact ``visual-reference-split/1``.

    Argumentos:
        path: caminho do JSON de partição.
    Retorna:
        partição tipada e estruturalmente válida.
    Levanta:
        ValueError: quando JSON, schema ou conteúdo forem inválidos.
    """
    try:
        root = mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON reference split {path}: {error.msg}.") from error
    raw_splits = mapping(root.get("splits"), f"{path}.splits")
    splits: dict[Split, tuple[str, ...]] = {}
    for split in Split:
        values = sequence(raw_splits, split.value, f"{path}.splits")
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError(f"{path}.splits.{split.value} must contain non-empty strings.")
        splits[split] = tuple(str(value) for value in values)
    return ReferenceSplit(
        schema_version=json_string(root, "schema_version", str(path)),
        reference_id=json_string(root, "reference_id", str(path)),
        source_manifest=json_string(root, "source_manifest", str(path)),
        source_manifest_sha256=json_string(root, "source_manifest_sha256", str(path)),
        partition_rule=json_string(root, "partition_rule", str(path)),
        splits=splits,
    )


# Compara a partição ao manifest byte a byte e semanticamente. Existe para
# impedir que um split antigo seja reutilizado após regeneração dos frames.
def validate_reference_split(
    partition: ReferenceSplit,
    manifest: ReferenceManifest,
    manifest_path: Path,
) -> None:
    """Valida digest, identidade, cobertura, exclusividade e ordem temporal.

    Argumentos:
        partition: artifact de split já carregado.
        manifest: manifest anotável correspondente.
        manifest_path: arquivo usado para calcular o digest registrado.
    Levanta:
        ValueError: quando a partição divergir ou permitir vazamento.
    """
    measured_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if measured_digest != partition.source_manifest_sha256:
        raise ValueError("reference split source manifest digest does not match the file.")
    if partition.reference_id != manifest.reference_id:
        raise ValueError("reference split and manifest have different reference_id values.")

    declared: dict[str, Split] = {}
    for split, identifiers in partition.splits.items():
        for identifier in identifiers:
            previous = declared.setdefault(identifier, split)
            if previous is not split:
                raise ValueError(f"sample {identifier!r} appears in more than one split.")
    manifest_by_id = {sample.sample_id: sample for sample in manifest.samples}
    if set(declared) != set(manifest_by_id):
        missing = sorted(set(manifest_by_id) - set(declared))
        unknown = sorted(set(declared) - set(manifest_by_id))
        raise ValueError(f"reference split coverage differs: missing={missing}, unknown={unknown}.")
    for identifier, split in declared.items():
        if manifest_by_id[identifier].split is not split:
            raise ValueError(f"sample {identifier!r} disagrees with its manifest split.")

    temporal_bounds: list[tuple[int, int, Split]] = []
    for split in Split:
        timestamps = [
            _source_timestamp(manifest_by_id[identifier].source_frame_id, identifier)
            for identifier in partition.splits[split]
        ]
        temporal_bounds.append((min(timestamps), max(timestamps), split))
    for (_left_min, left_max, left), (right_min, _right_max, right) in pairwise(
        temporal_bounds
    ):
        if left_max >= right_min:
            raise ValueError(
                f"temporal blocks {left.value!r} and {right.value!r} overlap or interleave."
            )


# Extrai o timestamp do frame de origem para verificar que blocos temporais
# inteiros, e não frames vizinhos intercalados, definem cada split.
def _source_timestamp(source_frame_id: str | None, sample_id: str) -> int:
    """Retorna o timestamp inteiro após ``:`` no ID do frame de origem."""
    if source_frame_id is None:
        raise ValueError(f"sample {sample_id!r} has no source_frame_id for temporal validation.")
    try:
        return int(source_frame_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(
            f"sample {sample_id!r} has an invalid temporal source_frame_id."
        ) from error


# Atalho seguro que carrega ambos os artifacts e executa toda validação.
def load_and_validate_reference_split(
    split_path: Path, manifest_path: Path
) -> ReferenceSplit:
    """Carrega a partição e confirma sua correspondência com o manifest."""
    partition = load_reference_split(split_path)
    manifest = load_reference_manifest(manifest_path)
    validate_reference_split(partition, manifest, manifest_path)
    return partition
