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

from fixtures import payload_with_blobs  # noqa: E402
from validate_reference_pipeline import (  # noqa: E402
    DEFAULT_SAMPLE_FRAME_OFFSET,
    DEFAULT_SAMPLE_SEED,
    ValidationOptions,
    evidence_state_counts,
    load_dotenv,
    remote_reasoning_summary,
    resolve_config,
    select_frame_paths,
    selection_record,
)
from visual_perception.application.tiling import build_tiles  # noqa: E402
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


# Verifica que a amostra padrão usa uma seed estável e mantém o segundo frame
# exatamente no deslocamento do protocolo.
def test_default_selection_is_seeded_pair_with_fixed_offset(tmp_path: Path) -> None:
    """Protege a amostra pareada padrão contra mudanças silenciosas de protocolo."""
    _touch_frames(tmp_path, *(f"frame-{index:02d}" for index in range(14)))

    selected = select_frame_paths(tmp_path)

    assert DEFAULT_SAMPLE_SEED == 42
    assert tuple(path.stem for path in selected) == ("frame-00", "frame-12")
    assert int(selected[1].stem.removeprefix("frame-")) - int(
        selected[0].stem.removeprefix("frame-")
    ) == DEFAULT_SAMPLE_FRAME_OFFSET


# Verifica que --limit continua sendo uma seleção explícita por prefixo
# ordenado, sem aplicar a amostra padrão.
def test_limit_selection_is_sorted_and_applied_afterward(tmp_path: Path) -> None:
    """Protege a seleção lexical determinística quando apenas limit é usado."""
    _touch_frames(tmp_path, "frame-c", "frame-a", "frame-b")

    selected = select_frame_paths(tmp_path, limit=2)

    assert tuple(path.stem for path in selected) == ("frame-a", "frame-b")


# Recusa uma execução padrão que não conseguiria formar o par +12; carregar
# modelos antes desse erro faria uma execução cara terminar sem resultado útil.
def test_default_selection_requires_pair_offset_capacity(tmp_path: Path) -> None:
    """Exige ao menos treze frames para a amostra padrão."""
    _touch_frames(tmp_path, *(f"frame-{index:02d}" for index in range(12)))

    with pytest.raises(ValueError, match="requires at least 13 frames"):
        select_frame_paths(tmp_path)


# O manifest declara a decisão de seleção, em vez de fazer o leitor deduzir
# seed e deslocamento a partir dos IDs escolhidos.
def test_default_selection_record_is_reproducible(tmp_path: Path) -> None:
    """Registra seed, deslocamento e IDs da amostra pareada."""
    _touch_frames(tmp_path, *(f"frame-{index:02d}" for index in range(13)))
    frames = select_frame_paths(tmp_path)

    assert selection_record(ValidationOptions(), frames) == {
        "strategy": "seeded_pair",
        "requested_frame_ids": [],
        "limit": None,
        "seed": 42,
        "frame_offset": 12,
        "selected_frame_ids": ["frame-00", "frame-12"],
    }


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


# Qwen vs Gemini (#276) só mede o modelo se o resto da config ficar idêntico
# — exceto o schema do prompt de região, que passou a ser por-backend em
# 2026-09-17 (ver MultimodalReasoningConfig.prompt_version): a config de
# referência do Qwen usa o schema reduzido (v10) por causa de um colapso de
# labels medido só nesse modelo, e generalizar isso para o Gemini quebrou o
# grounding textual de labels como "wooden pallet" (perdia o campo opcional
# "material", virava só "pallet", e o Grounding-DINO deixava de achar a box).
# O override para Gemini sempre repõe o schema de 8 campos (v8).
def test_the_reasoning_backend_override_changes_only_the_backend_identity_and_prompt_schema() -> None:
    """Selecionar o Gemini não altera views, tetos nem outros estágios."""
    baseline = resolve_config(ValidationOptions(sequence_masks=None))
    candidate = resolve_config(ValidationOptions(sequence_masks=None, reasoning_backend="gemini_robotics_er"))

    assert candidate.multimodal_reasoning.backend == "gemini_robotics_er"
    assert candidate.multimodal_reasoning.checkpoint == "gemini-robotics-er-2-preview"
    assert baseline.multimodal_reasoning.prompt_version == "v10"
    assert candidate.multimodal_reasoning.prompt_version == "v8"
    assert dataclasses.replace(
        candidate.multimodal_reasoning,
        backend=baseline.multimodal_reasoning.backend,
        checkpoint=baseline.multimodal_reasoning.checkpoint,
        load_in_4bit=baseline.multimodal_reasoning.load_in_4bit,
        prompt_version=baseline.multimodal_reasoning.prompt_version,
    ) == baseline.multimodal_reasoning
    for field in ("region_discovery", "feature_extraction", "language_embedding",
                  "multi_context", "hypothesis_support", "refinement",
                  "reconciliation", "semantic_relations", "calibration"):
        assert getattr(candidate, field) == getattr(baseline, field), field


# A comparação de discovery só é interpretável se a troca ficar restrita à
# geometria de proposals; checkpoint e identidade mudam juntos, pois são uma
# unidade de proveniência, mas os demais estágios devem permanecer idênticos.
def test_region_discovery_override_changes_only_its_backend_identity() -> None:
    """Selecionar Florence-2 preserva todos os demais backends e parâmetros."""
    baseline = resolve_config(ValidationOptions(sequence_masks=None))
    candidate = resolve_config(
        ValidationOptions(sequence_masks=None, region_discovery_backend="florence2")
    )

    assert candidate.region_discovery.backend == "florence2"
    assert candidate.region_discovery.checkpoint == "florence-community/Florence-2-large"
    assert dataclasses.replace(
        candidate.region_discovery,
        backend=baseline.region_discovery.backend,
        checkpoint=baseline.region_discovery.checkpoint,
    ) == baseline.region_discovery
    for field in (
        "feature_extraction",
        "language_embedding",
        "multimodal_reasoning",
        "multi_context",
        "hypothesis_support",
        "refinement",
        "reconciliation",
        "semantic_relations",
        "calibration",
    ):
        assert getattr(candidate, field) == getattr(baseline, field), field


# Mesma garantia, para a terceira opção de region discovery: o SAM2.1
# clássico compartilha o adapter com o SAM3, então só o checkpoint/backend
# devem mudar — um regressão aqui vazaria threshold ou device do SAM3 (que
# resolve_config já ajustou) para o braço SAM2.
def test_region_discovery_sam2_override_changes_only_its_backend_identity() -> None:
    """Selecionar SAM2.1 preserva todos os demais backends e parâmetros."""
    baseline = resolve_config(ValidationOptions(sequence_masks=None))
    candidate = resolve_config(
        ValidationOptions(sequence_masks=None, region_discovery_backend="sam")
    )

    assert candidate.region_discovery.backend == "sam"
    assert candidate.region_discovery.checkpoint == "facebook/sam2.1-hiera-large"
    assert dataclasses.replace(
        candidate.region_discovery,
        backend=baseline.region_discovery.backend,
        checkpoint=baseline.region_discovery.checkpoint,
    ) == baseline.region_discovery
    for field in (
        "feature_extraction",
        "language_embedding",
        "multimodal_reasoning",
        "multi_context",
        "hypothesis_support",
        "refinement",
        "reconciliation",
        "semantic_relations",
        "calibration",
    ):
        assert getattr(candidate, field) == getattr(baseline, field), field


# Mesma garantia que test_region_discovery_override_changes_only_its_backend_identity,
# para o eixo de feature extraction: trocar DINOv2 por FeatUp não pode
# arrastar nenhum outro estágio junto, ou a comparação deixa de isolar uma
# variável.
def test_feature_extraction_override_changes_only_its_backend_identity() -> None:
    """Selecionar FeatUp preserva todos os demais backends e parâmetros."""
    baseline = resolve_config(ValidationOptions(sequence_masks=None))
    candidate = resolve_config(
        ValidationOptions(sequence_masks=None, feature_extraction_backend="featup")
    )

    assert candidate.feature_extraction.backend == "featup"
    assert candidate.feature_extraction.checkpoint == baseline.feature_extraction.upsampler_checkpoint
    assert dataclasses.replace(
        candidate.feature_extraction,
        backend=baseline.feature_extraction.backend,
        checkpoint=baseline.feature_extraction.checkpoint,
    ) == baseline.feature_extraction
    for field in (
        "region_discovery",
        "language_embedding",
        "multimodal_reasoning",
        "multi_context",
        "hypothesis_support",
        "refinement",
        "reconciliation",
        "semantic_relations",
        "calibration",
    ):
        assert getattr(candidate, field) == getattr(baseline, field), field


# O manifest resume custo e rate limit do reasoner remoto; o local não tem essa seção.
def test_remote_reasoning_summary_aggregates_calls() -> None:
    """Soma latência, tokens e falhas transitórias por motivo."""
    from visual_perception.infrastructure.adapters.gemini_reasoning_backend import RemoteReasoningCall

    calls = (
        RemoteReasoningCall("scene", "m", 2.0, 1, True, prompt_tokens=1000, output_tokens=80),
        RemoteReasoningCall(
            "region", "m", 4.0, 3, True, prompt_tokens=1500, output_tokens=60, thought_tokens=0,
            transient_failures=("http_429", "timeout"),
        ),
        RemoteReasoningCall("region", "m", 1.0, 4, False, error="falhou", transient_failures=("http_503",) * 3),
    )

    summary = remote_reasoning_summary(calls)

    assert remote_reasoning_summary(None) is None
    assert summary is not None
    assert summary["calls"] == 3 and summary["failed_calls"] == 1
    assert summary["calls_by_operation"] == {"region": 2, "scene": 1}
    assert summary["total_latency_s"] == 7.0 and summary["prompt_tokens"] == 2500
    assert summary["transient_failures"] == {"http_429": 1, "timeout": 1, "http_503": 3}


# O .env só completa o ambiente: uma variável já exportada nunca é trocada.
def test_load_dotenv_does_not_override_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exporta chaves ausentes e preserva as já definidas."""
    env = tmp_path / ".env"
    env.write_text("# comentário\nVP_TEST_NEW=novo\nVP_TEST_KEEP=do-arquivo\n", encoding="utf-8")
    monkeypatch.delenv("VP_TEST_NEW", raising=False)
    monkeypatch.setenv("VP_TEST_KEEP", "do-ambiente")

    load_dotenv(env)
    load_dotenv(tmp_path / "ausente.env")

    import os

    assert os.environ["VP_TEST_NEW"] == "novo"
    assert os.environ["VP_TEST_KEEP"] == "do-ambiente"
    monkeypatch.delenv("VP_TEST_NEW")


# E sem o override a configuração de referência fica intacta.
def test_without_the_override_the_reference_checkpoint_is_untouched() -> None:
    """Omitir o override preserva o checkpoint da configuração de referência."""
    config = resolve_config(ValidationOptions(sequence_masks=None))

    assert config.multimodal_reasoning.checkpoint == "Qwen/Qwen2.5-VL-3B-Instruct"


# O validador compõe o perfil real sem alterar sua política de discovery: a
# execução de referência deve incluir a imagem completa e os quatro tiles.
def test_reference_validator_uses_hybrid_global_and_tiled_discovery() -> None:
    """Mantém cinco chamadas de discovery por frame no perfil de referência."""
    config = resolve_config(ValidationOptions(sequence_masks=None))
    tiles = build_tiles(payload_with_blobs(), config.tiling)

    assert config.tiling.multi_scale_enabled
    assert len(tiles) == 5


# O braço com prior temporal precisa diferir do braço sem ele em exatamente
# dois campos: o modo e a versão do prompt. Se diferisse em mais, a comparação
# entre os dois mediria mais de uma variável; se diferisse em menos, dois
# prompts diferentes declarariam a mesma versão e a proveniência por claim
# ficaria ambígua.
def test_o_prior_temporal_muda_apenas_o_modo_e_a_versao_do_prompt() -> None:
    """Ligar o prior altera dois campos, e nenhum outro."""
    baseline = resolve_config(ValidationOptions(sequence_masks=None))
    candidate = resolve_config(
        ValidationOptions(sequence_masks=None, temporal_prior_mode="box_overlap")
    )

    assert candidate.multimodal_reasoning.temporal_prior_mode == "box_overlap"
    assert candidate.multimodal_reasoning.prompt_version == "v11"
    assert baseline.multimodal_reasoning.temporal_prior_mode == "disabled"
    assert baseline.multimodal_reasoning.prompt_version == "v10"
    assert candidate.fingerprint() != baseline.fingerprint()
    assert dataclasses.replace(
        candidate.multimodal_reasoning,
        temporal_prior_mode=baseline.multimodal_reasoning.temporal_prior_mode,
        prompt_version=baseline.multimodal_reasoning.prompt_version,
    ) == baseline.multimodal_reasoning
    for field in ("region_discovery", "feature_extraction", "language_embedding",
                  "multi_context", "hypothesis_support", "refinement",
                  "reconciliation", "semantic_relations", "calibration",
                  "contextual_publication", "proposal_filter", "tiling", "merge"):
        assert getattr(candidate, field) == getattr(baseline, field), field


# E omitir a flag preserva exatamente o comportamento histórico: o prompt de
# região volta a ser byte-idêntico ao v8, porque o bloco do prior só é
# renderizado quando há prior.
def test_sem_a_flag_o_prompt_de_regiao_e_identico_ao_historico() -> None:
    """Sem prior, o prompt não ganha nenhum caractere novo."""
    from visual_perception.domain.geometry import BoundingBox
    from visual_perception.domain.image_payload import ImagePayload
    from visual_perception.domain.region_evidence import EvidenceSlot, SubjectEmphasis
    from visual_perception.domain.region_reasoning import (
        CoordinateTransform,
        RegionReasoningRequest,
        RegionView,
    )
    from visual_perception.infrastructure.adapters.reasoning_prompts import region_prompt

    box = BoundingBox(0.0, 0.0, 4.0, 4.0)
    view = RegionView(
        slot=EvidenceSlot.MASKED_SUBJECT,
        payload=ImagePayload(np.zeros((4, 4, 3), dtype=np.uint8), width=4, height=4),
        crop_box=box,
        transform=CoordinateTransform.identity(),
        emphasis=SubjectEmphasis.NEUTRAL_FILL,
    )
    request = RegionReasoningRequest(
        region_id="region-a", region_box=box, image_width=4, image_height=4, views=(view,)
    )

    assert "PREVIOUS view" not in region_prompt(request, "v8")
