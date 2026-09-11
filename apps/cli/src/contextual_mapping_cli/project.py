"""Descoberta e validação do estado local do projeto."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import tomllib

from .models import DatasetProfile, ValidationIssue

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


# Encapsula convenções de localização pertencentes à aplicação. Existe para
# que comandos e UI não reconstruam paths por concatenações divergentes.
class Project:
    """Estado local e recursos descobertos do repositório.

    Argumentos:
        root: raiz explícita; quando ausente, procura a partir do diretório atual.
    """

    # Inicializa a raiz validada usada por todos os workflows da aplicação.
    def __init__(self, root: Path | None = None) -> None:
        """Inicializa o projeto a partir de uma raiz explícita ou descoberta."""
        self.root = (root or self.discover_root(Path.cwd())).resolve()

    # Procura os marcadores estáveis do repositório nos diretórios ancestrais,
    # permitindo executar o console script a partir de qualquer subdiretório.
    @staticmethod
    def discover_root(start: Path) -> Path:
        """Encontra a raiz do repositório a partir de ``start``.

        Argumentos:
            start: diretório inicial da busca.
        Retorna:
            raiz que contém ``AGENTS.md``, ``Makefile`` e ``apps``.
        Levanta:
            FileNotFoundError: se nenhum ancestral tiver a estrutura esperada.
        """
        for candidate in (start, *start.parents):
            if (
                (candidate / "AGENTS.md").is_file()
                and (candidate / "Makefile").is_file()
                and (candidate / "apps").is_dir()
            ):
                return candidate
        raise FileNotFoundError("não foi possível localizar a raiz do contextual-3d-mapping")

    # Disponibiliza os pacotes irmãos enquanto o repositório ainda não possui
    # um workspace Python instalável único. É chamada antes de consumir APIs públicas.
    def activate_source_packages(self) -> None:
        """Adiciona as raízes de código do checkout ao import path atual."""
        paths = (
            self.root / "contracts",
            self.root / "datasets",
            self.root / "adapters" / "datasets",
            self.root / "modules" / "state-estimation" / "src",
            self.root / "modules" / "geometric-map" / "src",
            self.root / "modules" / "sensor-association" / "src",
            self.root / "modules" / "semantic-fusion" / "src",
            self.root / "apps" / "mapping-runtime" / "src",
        )
        for path in reversed(paths):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

    # Lista rosbags locais respeitando a raiz canônica de dados brutos.
    def available_bags(self) -> tuple[Path, ...]:
        """Retorna as rosbags locais em ordem estável."""
        return tuple(sorted((self.root / "datasets" / "raw").glob("**/*.bag")))

    # Lista artifacts de janela gerados por qualquer fluxo anterior.
    def available_windows(self) -> tuple[Path, ...]:
        """Retorna artifacts ``*-window.json`` em ordem estável."""
        return tuple(sorted((self.root / "artifacts").glob("*-window.json")))

    # Lista runs completos de visual-perception, ignorando diretórios parciais
    # que ainda não publicaram manifest.
    def available_runs(self) -> tuple[Path, ...]:
        """Retorna runs de percepção com ``manifest.json``."""
        roots = (
            self.root / "modules" / "visual-perception" / "benchmarks" / "results" / "samples",
            self.root / "artifacts" / "visual-runs" / "samples",
        )
        return tuple(sorted(path.parent for base in roots for path in base.glob("*/manifest.json")))

    # Carrega profiles declarativos pertencentes à composição da aplicação.
    # Novos datasets entram por configuração sem alterar os comandos públicos.
    def available_profiles(self) -> tuple[DatasetProfile, ...]:
        """Retorna os perfis de composição conhecidos pela aplicação."""
        profiles: list[DatasetProfile] = []
        for path in sorted((self.root / "apps" / "cli" / "configs").glob("*.toml")):
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
            try:
                profiles.append(
                    DatasetProfile(
                        profile_id=str(payload["profile_id"]),
                        bag=self._profile_path(str(payload["bag"])),
                        camera_topic=str(payload["camera_topic"]),
                        intrinsics=self._profile_path(str(payload["intrinsics"])),
                        extrinsics=self._profile_path(str(payload["extrinsics"])),
                        ground_truth=self._profile_path(str(payload["ground_truth"])),
                        sequence_masks=self._profile_path(str(payload["sequence_masks"])),
                        map_target=str(payload["map_target"]),
                        context_target=str(payload["context_target"]),
                    )
                )
            except KeyError as error:
                raise ValueError(f"campo ausente no profile {path}: {error.args[0]}") from error
        return tuple(profiles)

    # Resolve paths de profile dentro do checkout para evitar que configuração
    # versionada leia arquivos arbitrários da máquina.
    def _profile_path(self, value: str) -> Path:
        """Resolve um path relativo e garante que ele permaneça no repositório."""
        candidate = (self.root / value).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"path de profile sai do repositório: {value}") from error
        return candidate

    # Casa por caminho resolvido para impedir que outro arquivo homônimo herde
    # calibração ou máscaras do corridor-02 por engano.
    def profile_for(self, bag: Path) -> DatasetProfile | None:
        """Retorna o perfil compatível com ``bag``, quando existente."""
        resolved = bag.resolve()
        return next((item for item in self.available_profiles() if item.bag.resolve() == resolved), None)

    # Valida IDs antes que sejam interpolados em nomes de artifact ou variáveis
    # Make, evitando ambiguidades e saída fora do diretório esperado.
    @staticmethod
    def validate_segment_id(segment_id: str) -> str:
        """Valida e devolve uma identidade segura de segmento.

        Argumentos:
            segment_id: identidade fornecida pelo usuário.
        Retorna:
            a identidade validada.
        Levanta:
            ValueError: se houver caracteres inseguros.
        """
        if not _IDENTIFIER.fullmatch(segment_id):
            raise ValueError("segment-id deve usar apenas letras minúsculas, números, '.', '_' ou '-'")
        return segment_id

    # Produz diagnósticos sem alterar o checkout. Reúne apenas requisitos que
    # afetam os workflows oferecidos pela própria CLI.
    def validate(self) -> tuple[ValidationIssue, ...]:
        """Valida estrutura, dependências externas e artifacts descobertos."""
        issues: list[ValidationIssue] = []
        for directory in (
            self.root / "artifacts",
            self.root / "apps" / "map-explorer" / "web" / "public" / "maps",
        ):
            if not directory.is_dir():
                issues.append(ValidationIssue("erro", f"diretório ausente: {directory}", True))
        visual_python = self.root / "modules" / "visual-perception" / ".venv" / "bin" / "python"
        if not visual_python.exists():
            issues.append(ValidationIssue("erro", f"ambiente de visual-perception ausente: {visual_python}"))
        for executable in ("make", "docker", "npm"):
            if shutil.which(executable) is None:
                issues.append(ValidationIssue("aviso", f"executável opcional não encontrado: {executable}"))
        if not self.available_bags():
            issues.append(ValidationIssue("aviso", "nenhuma rosbag encontrada em datasets/raw"))
        for profile in self.available_profiles():
            for required in (
                profile.bag,
                profile.intrinsics,
                profile.extrinsics,
                profile.ground_truth,
                profile.sequence_masks,
            ):
                if not required.is_file():
                    issues.append(
                        ValidationIssue("erro", f"recurso ausente do perfil {profile.profile_id}: {required}")
                    )
        for window in self.available_windows():
            try:
                payload = json.loads(window.read_text(encoding="utf-8"))
                if not payload.get("keyframes"):
                    raise ValueError("sem keyframes")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                issues.append(ValidationIssue("erro", f"window inválida {window}: {error}"))
        return tuple(issues)

    # Repara somente diretórios derivados e ignorados pelo Git. Não instala
    # ferramentas nem modifica artifacts, mantendo a operação previsível.
    def repair(self) -> tuple[Path, ...]:
        """Cria diretórios locais seguros ausentes e retorna os paths criados."""
        created: list[Path] = []
        for directory in (
            self.root / "artifacts",
            self.root / "apps" / "map-explorer" / "web" / "public" / "maps",
        ):
            if not directory.is_dir():
                directory.mkdir(parents=True, exist_ok=True)
                created.append(directory)
        return tuple(created)
