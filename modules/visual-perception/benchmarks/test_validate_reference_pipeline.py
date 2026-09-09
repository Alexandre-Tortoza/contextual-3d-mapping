"""Testa a seleção e a auditoria reproduzíveis do validador real (#212)."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "src"))
sys.path.insert(0, str(_THIS_DIR.parent / "tests"))

from validate_reference_pipeline import (  # noqa: E402
    ValidationOptions,
    evidence_state_counts,
    resolve_config,
    select_frame_paths,
)
from visual_perception.domain.geometry import BoundingBox, Mask  # noqa: E402
from visual_perception.domain.region_evidence import (  # noqa: E402
    EvidenceSlot,
    EvidenceState,
    RegionEvidenceSlot,
)
from visual_perception.domain.regions import ObservedRegion  # noqa: E402
from visual_perception.domain.visual_observation import (  # noqa: E402
    SceneContext,
    VisualObservation,
)


# Executa o entrypoint como o usuário o invoca na documentação para impedir
# que os imports preparados pelo próprio pytest escondam um script quebrado.
def test_documented_validator_entrypoint_is_self_contained() -> None:
    """Confirma que ``python benchmarks/... --help`` resolve seus imports."""
    completed = subprocess.run(
        [sys.executable, str(_THIS_DIR / "validate_reference_pipeline.py"), "--help"],
        cwd=_THIS_DIR.parent,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--frame-id" in completed.stdout


# Cria arquivos PNG mínimos apenas pelo nome; a seleção não deve abrir nem
# interpretar conteúdo antes de validar IDs e ordem.
def _touch_frames(directory: Path, *frame_ids: str) -> None:
    """Cria entradas de frame vazias para testar somente a seleção."""
    directory.mkdir(exist_ok=True)
    for frame_id in frame_ids:
        (directory / f"{frame_id}.png").touch()


# Verifica a ordem vinculante dos IDs escolhidos para o teste real.
def test_explicit_frame_ids_preserve_requested_order(tmp_path: Path) -> None:
    """Garante que três IDs explícitos não sejam reordenados pelo filesystem."""
    _touch_frames(tmp_path, "corridor-02-017", "corridor-02-000", "corridor-02-008")

    selected = select_frame_paths(
        tmp_path,
        ("corridor-02-000", "corridor-02-008", "corridor-02-017"),
    )

    assert tuple(path.stem for path in selected) == (
        "corridor-02-000",
        "corridor-02-008",
        "corridor-02-017",
    )


# Verifica o caminho abreviado de seleção reproduzível por limite.
def test_default_selection_is_sorted_and_limit_is_applied_afterward(tmp_path: Path) -> None:
    """Protege a seleção lexical determinística quando apenas limit é usado."""
    _touch_frames(tmp_path, "frame-c", "frame-a", "frame-b")

    selected = select_frame_paths(tmp_path, limit=2)

    assert tuple(path.stem for path in selected) == ("frame-a", "frame-b")


@pytest.mark.parametrize(
    ("frame_ids", "match"),
    [(("missing",), "Unknown frame ids"), (("frame-a", "frame-a"), "duplicates")],
)
# Verifica que erros de identidade impedem uma execução parcial enganosa.
def test_invalid_selection_fails_before_model_loading(
    tmp_path: Path, frame_ids: tuple[str, ...], match: str
) -> None:
    """Rejeita ID ausente ou repetido sem permitir uma seleção parcial."""
    _touch_frames(tmp_path, "frame-a")

    with pytest.raises(ValueError, match=match):
        select_frame_paths(tmp_path, frame_ids)


# Monta uma observação mínima com os quatro estados/slots necessários para
# testar o resumo operacional sem executar modelos.
def _observation() -> VisualObservation:
    """Retorna uma observação com evidência disponível, ausente e falha."""
    mask_data = np.ones((2, 2), dtype=np.bool_)
    region = ObservedRegion(
        region_id="region-a",
        mask=Mask(mask_data, 2, 2),
        box=BoundingBox(0, 0, 2, 2),
        geometric_confidence=0.9,
        contributing_proposal_ids=("proposal-a",),
        evidence=(
            RegionEvidenceSlot(
                slot=EvidenceSlot.TIGHT_CROP,
                region_id="region-a",
                state=EvidenceState.MISSING,
                reason="slot disabled by configuration",
            ),
            RegionEvidenceSlot(
                slot=EvidenceSlot.CONTEXTUAL_CROP,
                region_id="region-a",
                state=EvidenceState.FAILED,
                reason="backend failure",
            ),
        ),
    )
    from fixtures import image_observation

    return VisualObservation(
        source=image_observation(width=2, height=2).source,
        image_width=2,
        image_height=2,
        scene_context=SceneContext(),
        regions=(region,),
        relations=(),
    )


# Confirma que cobertura desligada e erro de backend são contabilizados separadamente.
def test_manifest_summary_distinguishes_missing_from_failed_slots() -> None:
    """Mantém as causas operacionais distinguíveis no report por frame."""
    counts = evidence_state_counts(_observation())

    assert counts[EvidenceSlot.TIGHT_CROP.value][EvidenceState.MISSING.value] == 1
    assert counts[EvidenceSlot.CONTEXTUAL_CROP.value][EvidenceState.FAILED.value] == 1




# A comparação de backends (#218) só é interpretável se **um** fator mudar. O
# override precisa tocar o checkpoint e nada mais: se ele arrastasse junto o
# prompt, as views ou os tetos dos estágios contextuais, a diferença medida não
# poderia ser atribuída ao modelo.
def test_the_reasoning_checkpoint_override_changes_only_the_checkpoint() -> None:
    """Trocar o checkpoint do reasoner não altera nenhum outro campo da config."""
    baseline = resolve_config(ValidationOptions(sequence_masks=None))
    candidate = resolve_config(
        ValidationOptions(sequence_masks=None, reasoning_checkpoint="Qwen/Qwen3-VL-4B-Instruct")
    )

    assert candidate.multimodal_reasoning.checkpoint == "Qwen/Qwen3-VL-4B-Instruct"
    assert baseline.multimodal_reasoning.checkpoint != candidate.multimodal_reasoning.checkpoint
    assert dataclasses.replace(
        candidate.multimodal_reasoning, checkpoint=baseline.multimodal_reasoning.checkpoint
    ) == baseline.multimodal_reasoning
    for field in ("region_discovery", "feature_extraction", "language_embedding",
                  "multi_context", "hypothesis_support", "refinement",
                  "reconciliation", "semantic_relations", "calibration"):
        assert getattr(candidate, field) == getattr(baseline, field), field


# E sem o override a configuração de referência fica intacta.
def test_without_the_override_the_reference_checkpoint_is_untouched() -> None:
    """Omitir o override preserva o checkpoint da configuração de referência."""
    config = resolve_config(ValidationOptions(sequence_masks=None))

    assert config.multimodal_reasoning.checkpoint == "Qwen/Qwen2.5-VL-3B-Instruct"
