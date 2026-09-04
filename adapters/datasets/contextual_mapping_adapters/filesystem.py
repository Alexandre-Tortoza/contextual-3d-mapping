"""Resolução segura de artifacts locais para adapters de dataset."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from contextual_mapping_contracts import SourceArtifactReference
from contextual_mapping_datasets import DatasetManifest, raw_dataset_root


# Representa a fronteira de filesystem compartilhada por adapters concretos.
# Existe para concentrar a resolução segura entre um manifest e
# `datasets/raw/<dataset_id>`, sem incluir parsing específico de dataset.
class DatasetFilesystem:
    """Resolve artifacts declarados por um manifest sob sua raiz canônica."""

    # Inicializa a raiz local que pertence ao dataset e garante que ela existe
    # antes de um parser abrir artifacts. É chamada pelo adapter concreto no
    # momento de sua composição pela aplicação ou experimento.
    def __init__(self, repository_root: Path, manifest: DatasetManifest) -> None:
        self._manifest = manifest
        self._raw_root = raw_dataset_root(repository_root, manifest.dataset_id)
        if not self._raw_root.is_dir():
            raise FileNotFoundError(f"raw dataset directory does not exist: {self._raw_root}")

    # Expõe a raiz de leitura do dataset sem permitir que consumidores alterem
    # a convenção de layout. Parsers concretos usam esta propriedade para
    # localizar índices auxiliares declarados no manifest.
    @property
    def raw_root(self) -> Path:
        """Retorna a raiz local canônica do dataset."""
        return self._raw_root

    # Resolve uma referência relativa do manifest e constrói o contract de
    # artifact que será anexado às observações normalizadas pelo adapter.
    def source_artifact(self, relative_uri: str, media_type: str) -> SourceArtifactReference:
        """Resolve um artifact local existente e retorna sua referência pública.

        Argumentos:
            relative_uri: caminho relativo declarado no manifest.
            media_type: media type do artifact de origem.
        Retorna:
            referência `file:` estável para a observação canônica.
        Levanta:
            ValueError: se a referência não for um caminho relativo seguro.
            FileNotFoundError: se o artifact não existir na raiz do dataset.
        """
        path = self.resolve(relative_uri)
        if not path.is_file():
            raise FileNotFoundError(f"dataset artifact does not exist: {path}")
        return SourceArtifactReference(path.as_uri(), media_type)

    # Resolve um caminho relativo sob a raiz do dataset sem exigir que seja um
    # arquivo. Útil para parsers que precisam abrir diretórios ou índices, e
    # usado por `source_artifact` antes de validar um arquivo de origem.
    def resolve(self, relative_uri: str) -> Path:
        """Resolve uma referência relativa sem permitir saída da raiz local.

        Argumentos:
            relative_uri: caminho relativo declarado no manifest.
        Retorna:
            caminho absoluto confinado à raiz do dataset.
        Levanta:
            ValueError: se a URI usar scheme, host ou caminho absoluto.
        """
        parsed = urlsplit(relative_uri)
        relative_path = PurePosixPath(parsed.path)
        if (
            parsed.scheme
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or not parsed.path
            or relative_path.is_absolute()
            or any(part in {".", ".."} for part in relative_path.parts)
        ):
            raise ValueError("artifact URI must be a relative local path.")
        candidate = (self._raw_root / parsed.path).resolve()
        try:
            candidate.relative_to(self._raw_root.resolve())
        except ValueError as error:
            raise ValueError("artifact URI must remain inside the raw dataset directory.") from error
        return candidate
