"""Testes do schema de configuração de módulo validada (#157)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from visual_perception.application.execution_profile import research_quality_config
from visual_perception.config import (
    CalibrationConfig,
    HypothesisSupportConfig,
    ImageAreaConfig,
    ModuleConfig,
    MultiContextConfig,
    MultimodalReasoningConfig,
    QualityProfile,
    RefinementConfig,
    TilingConfig,
)
from visual_perception.domain.image_area import CircleArea, ImageAreaGeometry


# Confirma que a configuração default (sem nenhum campo explícito) é válida e usa o
# quality profile padrão esperado.
def test_minimal_config_validates() -> None:
    config = ModuleConfig()
    assert config.quality_profile is QualityProfile.RESEARCH_QUALITY


# Confirma que uma configuração com campos explícitos (orçamento de GPU, tiling
# multi-scale) também valida corretamente.
def test_complete_config_validates() -> None:
    tiling = TilingConfig(multi_scale_enabled=True, tile_grid="2x2")
    config = ModuleConfig(gpu_memory_budget_gb=12.0, tiling=tiling)
    assert config.tiling.multi_scale_enabled


# Protege o invariante de que o orçamento de memória de GPU deve ser positivo.
def test_rejects_non_positive_memory_budget() -> None:
    with pytest.raises(ValueError):
        ModuleConfig(gpu_memory_budget_gb=0)


# Protege a regra de negócio: o profile reduced_cost não pode ser combinado com tiling
# multi-scale (documentada em docs/architecture.md).
def test_rejects_incompatible_reduced_cost_multi_scale() -> None:
    tiling = TilingConfig(multi_scale_enabled=True)
    with pytest.raises(ValueError):
        ModuleConfig(quality_profile=QualityProfile.REDUCED_COST, tiling=tiling)


# Garante que um tile_grid mal formado é rejeitado cedo, na construção da config.
def test_rejects_invalid_tile_grid() -> None:
    with pytest.raises(ValueError):
        TilingConfig(tile_grid="not-a-grid")


# Confirma que a config sobrevive a um round-trip via to_dict/from_dict sem perda de
# informação — necessário para persistência e para o fingerprint do cache de estágio.
def test_config_round_trips_through_dict() -> None:
    config = ModuleConfig(tiling=TilingConfig(multi_scale_enabled=True, tile_grid="2x2"))
    restored = ModuleConfig.from_dict(config.to_dict())
    assert restored == config


# Garante que o fingerprint é determinístico: duas configs iguais produzem o mesmo
# fingerprint, condição necessária para o cache de estágio funcionar.
def test_fingerprint_is_stable_for_equal_configs() -> None:
    assert ModuleConfig().fingerprint() == ModuleConfig().fingerprint()


# Garante que configs diferentes produzem fingerprints diferentes, para que uma
# mudança de configuração invalide corretamente o cache.
def test_fingerprint_changes_when_config_changes() -> None:
    assert ModuleConfig().fingerprint() != ModuleConfig(gpu_memory_budget_gb=16.0).fingerprint()


# A seleção de views é o que decide qual evidência chega ao reasoner (#203).
# Um nome fora do vocabulário fechado de EvidenceSlot precisa falhar na
# construção da config, e não silenciosamente virar uma view a menos no run.
def test_region_views_rejects_a_slot_outside_the_closed_vocabulary() -> None:
    """Um slot de evidência desconhecido é rejeitado na configuração."""
    with pytest.raises(ValueError, match="unknown evidence slots"):
        MultimodalReasoningConfig(region_views=("foreground_dense", "panoramic_crop"))


# Interpretar uma região só a partir de contexto descreveria o entorno dela.
# A config recusa essa combinação em vez de deixar o run produzir labels que
# nenhuma evidência local sustenta.
def test_region_views_requires_at_least_one_foreground_slot() -> None:
    """Uma seleção só de contexto é rejeitada na configuração."""
    with pytest.raises(ValueError, match="at least one foreground slot"):
        MultimodalReasoningConfig(region_views=("contextual_crop", "scene_conditioned"))


# Repetir um slot mandaria a mesma imagem duas vezes ao VLM: custo puro, e uma
# ambiguidade sobre qual das duas produziu a resposta.
def test_region_views_rejects_a_repeated_slot() -> None:
    """Um slot repetido na seleção de views é rejeitado."""
    with pytest.raises(ValueError, match="must not repeat"):
        MultimodalReasoningConfig(region_views=("foreground_dense", "foreground_dense"))


# O default cobre foreground e contexto. A #212 chegou a removê-lo por hipótese
# e a medição reprovou: com os pixels de origem íntegros, o contextual_crop
# reduz o colapso de labels em corridor-02-002 de 0,50 para 0,30
# (docs/known-limitations.md, limitação 1).
def test_default_region_views_span_foreground_and_context() -> None:
    """A seleção default cobre foreground e contexto."""
    views = MultimodalReasoningConfig().region_views
    # O sujeito isolado que vai ao VLM é ``masked_subject``, e não
    # ``foreground_dense``: aquele é o slot do embedding denso, e usar o mesmo
    # recorte para os dois confundia a evidência do reasoner com a do DINO
    # (#202). O fundo dele é cinza, porque preto é a cor da vinheta do fisheye.
    assert "masked_subject" in views
    assert "tight_crop" in views
    assert "contextual_crop" in views


# O canal textual permanece ligado por default: medido em corridor-02-002,
# desligá-lo não foi o que corrigiu o colapso de labels (o canal visual foi), e
# custou um dominant_fraction levemente pior. O default só muda com medição a
# favor.
def test_default_scene_context_mode_is_context_assisted() -> None:
    """As claims de cena acompanham a região por default."""
    assert MultimodalReasoningConfig().scene_context_mode == "context_assisted"


# Garante que o modo local-first continua construível: ele é a única garantia
# estrutural de que nenhuma claim de cena vira label de região, e permanece
# disponível para a ablation mesmo não sendo o default.
def test_local_first_mode_is_configurable() -> None:
    """O modo local-first continua disponível por configuração."""
    config = MultimodalReasoningConfig(scene_context_mode="local_first")
    assert config.scene_context_mode == "local_first"


# Rejeita um modo desconhecido na fronteira de configuração: um typo precisa
# falhar na construção, e não produzir silenciosamente uma ablation diferente
# da que o experimento pensa estar rodando.
def test_unknown_scene_context_mode_is_rejected() -> None:
    """Um modo de contexto de cena desconhecido é recusado na construção."""
    with pytest.raises(ValueError, match="scene_context_mode"):
        MultimodalReasoningConfig(scene_context_mode="somewhat_local")


# Regressão da #241: ``asdict`` achata as geometrias de área em dicts, e
# ``from_dict`` só reconstruía o nível de topo — a config relida carregava dicts
# no lugar das geometrias e quebrava ao rasterizar. O caminho real passa por
# JSON (manifest do run), então o teste também passa.
def test_config_with_area_geometry_round_trips_through_json() -> None:
    """Uma config com círculo, polígono e sub-configs não-default volta igual do JSON."""
    config = ModuleConfig(
        image_area=ImageAreaConfig(
            valid_area=ImageAreaGeometry(circle=CircleArea(326.0, 244.0, 324.0)),
            ego_vehicle=ImageAreaGeometry(polygons=(((0.0, 400.0), (639.0, 400.0), (639.0, 479.0)),)),
        ),
        tiling=TilingConfig(multi_scale_enabled=True, tile_grid="2x2"),
        multi_context=MultiContextConfig(contextual_crop_enabled=True),
        hypothesis_support=HypothesisSupportConfig(slots=("tight_crop", "contextual_crop")),
        multimodal_reasoning=MultimodalReasoningConfig(region_views=("masked_subject", "contextual_crop")),
    )

    restored = ModuleConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    masks = restored.image_area.rasterize(640, 480)
    assert masks.valid_area is not None and masks.ego_vehicle is not None


# ``ImageAreaConfig`` não aceita mais dicts no lugar das geometrias, que era o
# que deixava a config corrompida passar.
def test_image_area_config_rejects_non_geometry_values() -> None:
    """Um dict no lugar de geometria falha na construção."""
    with pytest.raises(TypeError, match="valid_area"):
        ImageAreaConfig(valid_area={"circle": None})  # type: ignore[arg-type]


# O artifact de calibração é conferido na construção, e o fingerprint não
# depende mais do arquivo continuar existindo: antes ele relia o disco a cada
# chamada e levantava FileNotFoundError longe da origem.
def test_calibration_artifact_is_validated_once_at_construction(tmp_path: Path) -> None:
    """Artifact ausente falha na construção; o fingerprint não faz I/O depois dela."""
    with pytest.raises(ValueError, match="does not exist"):
        CalibrationConfig(enabled=True, artifact_path=str(tmp_path / "missing.json"))

    artifact = tmp_path / "calibration.json"
    artifact.write_text("{}", encoding="utf-8")
    config = ModuleConfig(calibration=CalibrationConfig(enabled=True, artifact_path=str(artifact)))
    fingerprint = config.fingerprint()
    artifact.unlink()

    assert config.fingerprint() == fingerprint
    assert config.calibration.artifact_digest is not None


# O digest gravado na config permite detectar que o artifact mudou desde que
# ela foi escrita; reler a config reproduziria outra calibração em silêncio.
def test_rereading_a_config_whose_calibration_artifact_changed_fails(tmp_path: Path) -> None:
    """Um artifact alterado desde a serialização faz ``from_dict`` falhar."""
    artifact = tmp_path / "calibration.json"
    artifact.write_text("{}", encoding="utf-8")
    payload = ModuleConfig(
        calibration=CalibrationConfig(enabled=True, artifact_path=str(artifact))
    ).to_dict()
    artifact.write_text('{"changed": true}', encoding="utf-8")

    with pytest.raises(ValueError, match="changed since"):
        ModuleConfig.from_dict(payload)


# Regressão da #241: por default o crop contextual estava desligado e ainda
# assim era pedido pelo suporte de hipótese e pelo refinamento — um terço dos
# sinais saía sempre indisponível.
@pytest.mark.parametrize(
    ("field_name", "override"),
    [
        ("hypothesis_support.slots", {"hypothesis_support": HypothesisSupportConfig(slots=("contextual_crop",))}),
        (
            "refinement.escalation_views",
            {"refinement": RefinementConfig(escalation_views=("masked_subject", "contextual_crop"))},
        ),
    ],
)
def test_requesting_a_disabled_evidence_slot_fails(field_name: str, override: dict[str, object]) -> None:
    """Pedir um slot desligado em ``multi_context`` falha nomeando os dois campos."""
    with pytest.raises(ValueError, match=rf"{re.escape(field_name)}.*multi_context"):
        ModuleConfig(**override)  # type: ignore[arg-type]


# Os dois conjuntos de defaults precisam satisfazer a invariante: o do módulo
# (crop contextual desligado) e o do perfil real (ligado, e consumido).
def test_module_defaults_and_real_profile_request_only_enabled_slots() -> None:
    """Default e perfil real constroem sem pedir slot desligado."""
    default = ModuleConfig()
    real = research_quality_config(multi_scale_justified=False, real_backends=True)

    assert "contextual_crop" not in default.hypothesis_support.slots
    assert "contextual_crop" in real.hypothesis_support.slots
    assert "contextual_crop" in real.refinement.escalation_views
