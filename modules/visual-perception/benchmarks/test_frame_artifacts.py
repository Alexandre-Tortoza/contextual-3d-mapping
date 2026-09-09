"""Testes do layout de artifacts por frame e da imutabilidade da entrada (#212).

Cobrem a lacuna que ``test_validate_reference_pipeline.py`` deixava: nada
exercitava ``run_validation`` de ponta a ponta, então uma regressão no que é
escrito em disco só apareceria num run real de GPU. Aqui tudo roda com fakes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "tests"))

from PIL import Image  # noqa: E402

from fixtures_ports import default_ports  # noqa: E402
from inspect_region import load_frame, write_region_inspection  # noqa: E402
from validate_reference_pipeline import ValidationOptions, run_validation  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager  # noqa: E402
from visual_perception.application.pipeline import PerceptionPorts  # noqa: E402
from visual_perception.application.region_views import build_region_views  # noqa: E402
from visual_perception.config import ModuleConfig, RegionDiscoveryConfig  # noqa: E402
from visual_perception.domain.geometry import Mask  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.regions import LocalRegionProposal  # noqa: E402

_FRAME_ID = "synthetic-000"

#: Camadas que todo frame bem-sucedido precisa produzir. Um artifact que sumir
#: daqui deixa uma pergunta de diagnóstico sem resposta, então a lista é a
#: especificação do layout e não uma conveniência do teste.
_EXPECTED_ARTIFACTS = (
    "observation.json",
    "diagnostics.json",
    "raw.png",
    "proposals.png",
    "regions-masks.png",
    "regions-boxes.png",
    "regions-labels.png",
    "regions-overlay.png",
)

#: Máscaras de área, escritas só quando a sequência declara geometria. Ficam
#: fora de _EXPECTED_ARTIFACTS porque a ausência delas é informação: um frame
#: sem geometria declarada não teve exclusão nenhuma aplicada.
_AREA_ARTIFACTS = ("ego-mask.png", "valid-area-mask.png")


# Gera um frame sintético com blobs separados, para o fake discoverer encontrar
# várias regiões sem depender de nenhuma imagem real do dataset. A resolução
# acompanha a dos frames reais do corridor-02 porque a máscara do ego-veículo é
# definida por uma linha absoluta: num frame mais baixo ela não cobriria nada e
# o teste de mascaramento não exercitaria o caminho destrutivo.
def _write_frame(frames_dir: Path, width: int = 640, height: int = 480) -> np.ndarray:
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    pixels[20:60, 20:70] = (200, 30, 30)
    pixels[70:100, 90:140] = (30, 60, 200)
    frames_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(frames_dir / f"{_FRAME_ID}.png")
    return pixels


# Injeta os fakes preservando a assinatura de create_perception_ports; é para
# isso que o seam de run_validation é uma factory e não um PerceptionPorts pronto.
def _fake_ports(config: ModuleConfig, lifecycle: ModelLifecycleManager) -> PerceptionPorts:
    return default_ports()


# Roda a validação sintética e devolve o diretório do frame produzido.
def _run(tmp_path: Path, **options: object) -> Path:
    frames_dir = tmp_path / "frames-in"
    _write_frame(frames_dir)
    out_dir = run_validation(
        ValidationOptions(
            frames_dir=frames_dir,
            results_dir=tmp_path / "out",
            sequence_masks=None,
            **options,  # type: ignore[arg-type]
        ),
        ports_factory=_fake_ports,
    )
    return out_dir / "frames" / _FRAME_ID


# O layout é o produto desta mudança: cada camada responde uma pergunta
# diferente sobre onde o erro nasceu, e uma delas faltando reabre a ambiguidade
# que a separação existe para fechar.
def test_run_writes_every_layer_of_the_frame_layout(tmp_path: Path) -> None:
    """Um run sintético produz todas as camadas de inspeção do frame."""
    frame_dir = _run(tmp_path)

    for artifact in _EXPECTED_ARTIFACTS:
        assert (frame_dir / artifact).is_file(), f"missing {artifact}"


# A regra inegociável desta entrega: os pixels de origem são imutáveis. O
# validator antes zerava a faixa do ego-veículo antes do SAM e redesenhava o
# overlay sobre a imagem já destruída, então nada no artifact revelava a
# alteração.
def test_raw_pixels_reach_the_pipeline_unmodified(tmp_path: Path) -> None:
    """Sem mascaramento explícito, o pipeline recebe exatamente os pixels de origem."""
    frames_dir = tmp_path / "frames-in"
    source_pixels = _write_frame(frames_dir)
    out_dir = run_validation(
        ValidationOptions(
            frames_dir=frames_dir, results_dir=tmp_path / "out", sequence_masks=None
        ),
        ports_factory=_fake_ports,
    )
    frame_dir = out_dir / "frames" / _FRAME_ID

    saved_raw = np.array(Image.open(frame_dir / "raw.png").convert("RGB"))
    diagnostics = json.loads((frame_dir / "diagnostics.json").read_text())

    assert np.array_equal(saved_raw, source_pixels)
    assert diagnostics["pipeline_input_identical_to_raw"] is True
    # A entrada só é persistida quando difere: um segundo PNG idêntico por frame
    # não informaria nada.
    assert not (frame_dir / "pipeline-input.png").is_file()
    assert diagnostics["ego_vehicle_mask_applied"] is False


# A geometria declarada da sequência é aplicada *dentro* do pipeline, sobre
# máscaras. O caminho destrutivo que existia até a #212 — zerar a faixa do rig
# antes do SAM — foi removido: nenhuma opção do harness altera mais os pixels.
def test_declared_geometry_is_applied_without_touching_the_pixels(tmp_path: Path) -> None:
    """Com geometria declarada, a exclusão é aplicada e os pixels ficam intactos."""
    frames_dir = tmp_path / "frames-in"
    source_pixels = _write_frame(frames_dir)
    masks = tmp_path / "corridor-02.json"
    masks.write_text(
        json.dumps(
            {
                "image_width": 640,
                "image_height": 480,
                "valid_area": {"circle": {"cx": 326.0, "cy": 244.0, "r": 324.0}},
                "ego_vehicle": {"polygons": [[[0, 400], [639, 400], [639, 479], [0, 479]]]},
            }
        )
    )
    out_dir = run_validation(
        ValidationOptions(
            frames_dir=frames_dir, results_dir=tmp_path / "out", sequence_masks=masks
        ),
        ports_factory=_fake_ports,
    )
    frame_dir = out_dir / "frames" / _FRAME_ID

    saved_raw = np.array(Image.open(frame_dir / "raw.png").convert("RGB"))
    diagnostics = json.loads((frame_dir / "diagnostics.json").read_text())

    assert np.array_equal(saved_raw, source_pixels)
    assert diagnostics["pipeline_input_identical_to_raw"] is True
    assert not (frame_dir / "pipeline-input.png").is_file()
    assert diagnostics["ego_vehicle_mask_applied"] is True
    assert diagnostics["valid_fisheye_mask"] == "applied"
    assert (frame_dir / "ego-mask.png").is_file()
    assert (frame_dir / "valid-area-mask.png").is_file()


# Aplicar uma geometria medida para outra resolução rejeitaria o frame inteiro
# sem deixar rastro. A execução falha alto em vez de produzir zero regiões.
def test_geometry_measured_for_another_resolution_fails_loudly(tmp_path: Path) -> None:
    """Uma geometria de resolução diferente interrompe o run com mensagem clara."""
    frames_dir = tmp_path / "frames-in"
    _write_frame(frames_dir)
    masks = tmp_path / "wrong.json"
    masks.write_text(
        json.dumps(
            {
                "image_width": 1280,
                "image_height": 720,
                "valid_area": {"circle": {"cx": 640.0, "cy": 360.0, "r": 300.0}},
            }
        )
    )

    with pytest.raises(SystemExit, match="1280x720"):
        run_validation(
            ValidationOptions(
                frames_dir=frames_dir, results_dir=tmp_path / "out", sequence_masks=masks
            ),
            ports_factory=_fake_ports,
        )


# Sem geometria declarada a ausência continua explícita: nada é inferido por
# limiar de luminância, e o artifact diz que não houve exclusão em vez de
# sugerir que houve.
def test_absent_geometry_is_declared_not_invented(tmp_path: Path) -> None:
    """Sem geometria declarada, a exclusão é reportada como indisponível."""
    frame_dir = _run(tmp_path)
    diagnostics = json.loads((frame_dir / "diagnostics.json").read_text())

    assert diagnostics["valid_fisheye_mask"] == "unavailable"
    assert diagnostics["ego_vehicle_mask_applied"] is False
    assert not (frame_dir / "valid-area-mask.png").is_file()
    assert not (frame_dir / "ego-mask.png").is_file()


# Descobre duas proposals quase idênticas sobre o mesmo blob, para exercitar o
# merge geométrico. O FakeRegionDiscoverer rotula componentes conectados e por
# construção não consegue produzir duas proposals sobrepostas — daí o duplo
# local, e daí o seam de ports precisar ser uma factory.
class _TwoProposalDiscoverer:
    """Emite duas proposals sobrepostas o bastante para o merge uni-las."""

    def discover(
        self, image: ImagePayload, config: RegionDiscoveryConfig
    ) -> tuple[LocalRegionProposal, ...]:
        """Retorna duas proposals com IoU acima do limiar de merge."""
        first = np.zeros((image.height, image.width), dtype=np.bool_)
        first[20:60, 20:70] = True
        second = first.copy()
        second[59, 69] = False
        return tuple(
            LocalRegionProposal(
                local_id=f"overlapping-{index}",
                mask=Mask(data, image.width, image.height),
                box=Mask(data, image.width, image.height).bounding_box(),
                geometric_confidence=0.9,
                source="test_two_proposal_discoverer",
            )
            for index, data in enumerate((first, second))
        )


# A separação que motivou expor as proposals: com um único overlay não havia
# como saber se a duplicação vinha do SAM ou de o merge não ter unido.
def test_proposal_and_region_stages_are_separately_observable(tmp_path: Path) -> None:
    """Duas proposals fundidas em uma região aparecem como dois estágios distintos."""
    frames_dir = tmp_path / "frames-in"
    _write_frame(frames_dir)

    def _merging_ports(config: ModuleConfig, lifecycle: ModelLifecycleManager) -> PerceptionPorts:
        base = default_ports()
        return PerceptionPorts(
            region_discoverer=_TwoProposalDiscoverer(),
            feature_extractor=base.feature_extractor,
            language_encoder=base.language_encoder,
            multimodal_reasoner=base.multimodal_reasoner,
        )

    out_dir = run_validation(
        ValidationOptions(
            frames_dir=frames_dir, results_dir=tmp_path / "out", sequence_masks=None
        ),
        ports_factory=_merging_ports,
    )
    frame_dir = out_dir / "frames" / _FRAME_ID
    diagnostics = json.loads((frame_dir / "diagnostics.json").read_text())

    assert diagnostics["proposal_count"] == 2
    assert diagnostics["region_count"] == 1
    assert diagnostics["regions_with_multiple_proposals"] == 1
    assert diagnostics["merged_proposal_count"] == 2
    # Os dois estágios precisam render izar imagens distintas: se fossem iguais,
    # o artifact separado não estaria mostrando nada de novo.
    proposals_png = (frame_dir / "proposals.png").read_bytes()
    regions_png = (frame_dir / "regions-overlay.png").read_bytes()
    assert proposals_png != regions_png


# O diagnóstico precisa carregar a métrica de colapso para um run real ser
# avaliável sem abrir imagem nenhuma.
def test_diagnostics_report_mode_collapse_fields(tmp_path: Path) -> None:
    """O diagnóstico do frame traz label dominante e sua fração."""
    frame_dir = _run(tmp_path)
    diagnostics = json.loads((frame_dir / "diagnostics.json").read_text())

    assert "dominant_label" in diagnostics["mode_collapse"]
    assert 0.0 <= diagnostics["mode_collapse"]["dominant_fraction"] <= 1.0
    assert diagnostics["scene_context_mode"] == "context_assisted"


# A garantia que faz a inspeção sob demanda valer: as views materializadas
# depois têm que ser exatamente as que o reasoner recebeu. Se o benchmark
# reimplementasse o crop, elas divergiriam e o artifact mentiria.
def test_region_views_materialized_on_demand_match_the_pipeline(tmp_path: Path) -> None:
    """As views reconstruídas do disco são idênticas às de build_region_views."""
    frame_dir = _run(tmp_path)
    snapshot = load_frame(frame_dir)
    region = snapshot.observation.regions[0]

    written = write_region_inspection(snapshot, region.region_id, tmp_path / "inspect")
    expected = build_region_views(
        snapshot.observation.regions, snapshot.payload, snapshot.config
    )[region.region_id]
    by_slot = {view.slot.value: view for view in expected}

    for slot_value, filename in written.items():
        if slot_value not in by_slot:
            continue
        saved = np.array(Image.open(tmp_path / "inspect" / filename).convert("RGB"))
        assert np.array_equal(saved, by_slot[slot_value].payload.pixels), slot_value
    assert (tmp_path / "inspect" / "semantics.json").is_file()


# Um slot desabilitado na config não produz arquivo, e essa ausência é
# informação: é a mesma doutrina de ``missing`` versus ``failed`` que o contract
# de evidência já usa.
def test_disabled_view_slot_produces_no_file_and_is_recorded(tmp_path: Path) -> None:
    """Uma view desabilitada fica ausente do disco e registrada como ausente."""
    frame_dir = _run(tmp_path, context_profile="baseline")
    snapshot = load_frame(frame_dir)
    region = snapshot.observation.regions[0]

    write_region_inspection(snapshot, region.region_id, tmp_path / "inspect")
    semantics = json.loads((tmp_path / "inspect" / "semantics.json").read_text())

    assert not (tmp_path / "inspect" / "contextual.png").is_file()
    assert "contextual_crop" in semantics["views_missing"]
    assert "foreground_dense" in semantics["views_present"]


# Construída não é o mesmo que entregue: ``multi_context`` decide o que é
# extraído e ``region_views`` decide o que chega ao reasoner. Um artifact que só
# listasse as views construídas sugeriria que o modelo viu imagens que nunca
# recebeu — a mesma classe de erro do overlay que mostrava confiança geométrica
# como se fosse semântica.
def test_semantics_distinguishes_views_built_from_views_delivered(tmp_path: Path) -> None:
    """O resumo separa as views construídas das que o reasoner recebeu."""
    frame_dir = _run(tmp_path)
    snapshot = load_frame(frame_dir)
    region = snapshot.observation.regions[0]

    write_region_inspection(snapshot, region.region_id, tmp_path / "inspect")
    semantics = json.loads((tmp_path / "inspect" / "semantics.json").read_text())

    delivered = set(semantics["views_delivered_to_reasoner"])
    built = set(semantics["views_built"])
    assert delivered <= built
    assert delivered == set(snapshot.config.multimodal_reasoning.region_views) & built
    # O perfil de pesquisa extrai a view de cena inteira, mas ela nunca está em
    # region_views: é evidência de embedding, não imagem para o reasoner. Se as
    # duas listas fossem iguais, o teste não estaria medindo nada.
    assert "scene_conditioned" in built
    assert "scene_conditioned" not in delivered



# Os vetores precisam sair do processo por referência: até a #217 eles eram
# calculados — 121 chamadas de encoder por frame na configuração real — e
# descartados dentro do pipeline, deixando cada ``artifact_ref`` de slot
# apontando para nada. Este teste é o que impede aquele estado de voltar em
# silêncio.
def test_embeddings_are_persisted_and_resolve_the_slot_references(tmp_path: Path) -> None:
    """Cada ``artifact_ref`` de slot disponível resolve para um vetor gravado."""
    frame_dir = _run(tmp_path)

    assert (frame_dir / "embeddings.npz").is_file()
    stored = np.load(frame_dir / "embeddings.npz")
    observation = json.loads((frame_dir / "observation.json").read_text())
    referenced = {
        slot["artifact_ref"]
        for region in observation["regions"]
        for slot in region["evidence"]
        if slot["state"] == "available" and slot["artifact_ref"] is not None
    }

    assert referenced
    assert referenced <= set(stored.files)
    for name in stored.files:
        assert np.isfinite(stored[name]).all()


# O que a #214/#205/#206 acrescentaram precisa aparecer no diagnóstico do
# frame: um estágio cujo efeito não é contável no artifact não é comparável
# entre runs, e a comparação é a única forma de decidir se ele vale o custo.
def test_the_frame_diagnostics_report_the_contextual_stages(tmp_path: Path) -> None:
    """O diagnóstico do frame conta sinais, reconciliação, grupos e relações."""
    frame_dir = _run(tmp_path)

    diagnostics = json.loads((frame_dir / "diagnostics.json").read_text())
    contextual = diagnostics["contextual"]

    assert set(contextual) >= {
        "signal_statuses",
        "regions_with_unsupported_primary",
        "regions_with_ambiguous_identity",
        "region_kind_contradictions",
        "reconciled_regions",
        "distinct_raw_labels",
        "distinct_canonical_concepts",
        "entity_groups",
        "regions_in_entity_groups",
        "semantic_relations",
        "geometric_relations",
        "abstained_claims",
        "unscored_claims",
    }
    assert diagnostics["layout_version"] == "frames/2"
