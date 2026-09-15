"""Testes de descoberta e validação do estado do projeto pela CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from contextual_mapping_cli.project import Project


# Materializa somente os marcadores necessários para exercitar a descoberta
# sem depender do checkout real do teste.
def _project_root(tmp_path: Path) -> Path:
    """Cria uma raiz mínima reconhecida pela aplicação."""
    (tmp_path / "AGENTS.md").write_text("# teste", encoding="utf-8")
    (tmp_path / "Makefile").write_text("test:\n", encoding="utf-8")
    (tmp_path / "apps" / "nested").mkdir(parents=True)
    return tmp_path


# O console script pode ser chamado de qualquer subdiretório do repositório.
def test_discover_root_sobe_ate_os_marcadores(tmp_path: Path) -> None:
    """Confere a descoberta a partir de um diretório interno."""
    root = _project_root(tmp_path)
    assert Project.discover_root(root / "apps" / "nested") == root


# IDs entram em paths e variáveis Make, portanto separadores de diretório não
# podem atravessar a fronteira da aplicação.
def test_validate_segment_id_rejeita_path(tmp_path: Path) -> None:
    """Confere a gramática segura de identidades de segmento."""
    project = Project(_project_root(tmp_path))
    assert project.validate_segment_id("warehouse-01.2") == "warehouse-01.2"
    with pytest.raises(ValueError, match="segment-id"):
        project.validate_segment_id("../fora")


# A reparação é deliberadamente estreita: apenas diretórios ignorados e
# derivados são criados, sem instalar ferramentas ou inventar dados brutos.
def test_repair_cria_apenas_diretorios_derivados(tmp_path: Path) -> None:
    """Confere os destinos seguros da reparação estrutural."""
    project = Project(_project_root(tmp_path))
    created = project.repair()
    assert project.root / "artifacts" in created
    assert (project.root / "apps" / "map-explorer" / "web" / "public" / "maps").is_dir()
    assert not (project.root / "datasets" / "raw").exists()


# Profiles são configuração de composição e precisam ser adicionáveis sem
# alterar a implementação dos comandos.
def test_available_profiles_carrega_toml_relativo_a_raiz(tmp_path: Path) -> None:
    """Confere parsing e resolução confinada dos paths do profile."""
    root = _project_root(tmp_path)
    configs = root / "apps" / "cli" / "configs"
    configs.mkdir(parents=True)
    (configs / "warehouse.toml").write_text(
        "\n".join(
            (
                'profile_id = "warehouse"',
                'bag = "datasets/raw/warehouse/run.bag"',
                'camera_topic = "/rgb"',
                'intrinsics = "datasets/raw/warehouse/intrinsics.yaml"',
                'extrinsics = "datasets/raw/warehouse/extrinsics.yaml"',
                'ground_truth = "datasets/raw/warehouse/gt.txt"',
                'sequence_masks = "datasets/raw/warehouse/masks.json"',
                'map_target = "warehouse-map"',
                'context_target = "warehouse-context"',
            )
        ),
        encoding="utf-8",
    )
    profile = Project(root).available_profiles()[0]
    assert profile.profile_id == "warehouse"
    assert profile.bag == root / "datasets" / "raw" / "warehouse" / "run.bag"
