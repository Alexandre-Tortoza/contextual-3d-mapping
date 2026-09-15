"""Testes do layout de artifacts por frame e da imutabilidade da entrada (#212).

Cobrem a lacuna que ``test_validate_reference_pipeline.py`` deixava: nada
exercitava ``run_validation`` de ponta a ponta, então uma regressão no que é
escrito em disco só apareceria num run real de GPU. Aqui tudo roda com fakes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "tests"))

from contextual_mapping_adapters import ExtractedFrameProvenance, write_frame_provenance  # noqa: E402
from PIL import Image  # noqa: E402

from fixtures import image_observation, payload_with_blobs  # noqa: E402
from fixtures_ports import default_ports  # noqa: E402
from frame_artifacts import FrameInputs, write_frame_artifacts  # noqa: E402
from inspect_region import load_frame, write_region_inspection  # noqa: E402
from validate_reference_pipeline import ValidationOptions, run_validation  # noqa: E402
from visual_perception.application.lifecycle import ModelLifecycleManager  # noqa: E402
from visual_perception.application.observation_diagnostics import diagnose_observation  # noqa: E402
from visual_perception.application.pipeline import PerceptionPorts, run_canonical_pipeline  # noqa: E402
from visual_perception.application.region_views import build_region_views  # noqa: E402
from visual_perception.config import ModuleConfig, RegionDiscoveryConfig  # noqa: E402
from visual_perception.domain.geometry import Mask  # noqa: E402
from visual_perception.domain.grounding import (  # noqa: E402
    GroundingPrediction,
    GroundingStatus,
    SemanticGrounding,
)
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.regions import LocalRegionProposal, primary_label_claim  # noqa: E402
from visual_perception.infrastructure.embedding_archive import (  # noqa: E402
    UnresolvableEmbeddingRefError,
    resolve_embedding_vector,
)
from visual_perception.infrastructure.fakes.fake_region_discoverer import FakeRegionDiscoverer  # noqa: E402

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


# Garante que o full debug preserva as cinco passadas reais de discovery e as
# três views enviadas ao reasoner. Sem este teste, uma mudança no writer poderia
# voltar a deixar tiles e recortes disponíveis apenas em memória.
def test_full_debug_writes_discovery_passes_and_reasoner_views(tmp_path: Path) -> None:
    """Grava imagem/propostas globais e tiled, mais as três views por região."""
    frame_dir = _run(tmp_path)
    discovery = frame_dir / "DEBUG" / "discovery"
    expected_discovery = {
        f"{scale}-{tile}-{kind}.png"
        for scale, tile in (("full", "whole"), ("tile", "r0c0"), ("tile", "r0c1"),
                            ("tile", "r1c0"), ("tile", "r1c1"))
        for kind in ("input", "proposals")
    }
    assert expected_discovery <= {path.name for path in discovery.iterdir()}

    region_directories = [path for path in (frame_dir / "DEBUG" / "regions").iterdir() if path.is_dir()]
    assert region_directories
    assert {
        "masked-subject.png", "tight-crop.png", "contextual-crop.png",
    } <= {path.name for path in region_directories[0].iterdir()}


# Gera um frame sintético com blobs separados, para o fake discoverer encontrar
# várias regiões sem depender de nenhuma imagem real do dataset. A resolução
# acompanha a dos frames reais do corridor-02 porque a máscara do ego-veículo é
# definida por uma linha absoluta: num frame mais baixo ela não cobriria nada e
# o teste de mascaramento não exercitaria o caminho destrutivo.
def _write_frame(
    frames_dir: Path,
    width: int = 640,
    height: int = 480,
    *,
    frame_id: str = _FRAME_ID,
    sequence_index: int = 0,
) -> np.ndarray:
    pixels = np.full((height, width, 3), 255, dtype=np.uint8)
    pixels[20:60, 20:70] = (200, 30, 30)
    pixels[70:100, 90:140] = (30, 60, 200)
    frames_dir.mkdir(parents=True, exist_ok=True)
    image_path = frames_dir / f"{frame_id}.png"
    Image.fromarray(pixels).save(image_path)
    # Todo frame extraído carrega a proveniência da mensagem de origem; o
    # validador recusa frames sem ela (#239).
    write_frame_provenance(
        image_path,
        ExtractedFrameProvenance(
            recording_id="synthetic",
            topic="/camera/image_raw",
            frame_id="camera_optical_frame",
            timestamp_ns=1_000_000_000 + sequence_index * 100_000_000,
            sequence_index=sequence_index,
        ),
    )
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

    embeddings_path = frame_dir / "embeddings.npz"
    assert embeddings_path.is_file()
    observation = json.loads((frame_dir / "observation.json").read_text())
    referenced = {
        slot["artifact_ref"]
        for region in observation["regions"]
        for slot in region["evidence"]
        if slot["state"] == "available" and slot["artifact_ref"] is not None
    }

    assert referenced
    for ref in referenced:
        vector = resolve_embedding_vector(embeddings_path, ref)
        assert vector
        assert all(math.isfinite(component) for component in vector)

    with pytest.raises(UnresolvableEmbeddingRefError):
        resolve_embedding_vector(embeddings_path, "does-not-exist")


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
    assert diagnostics["layout_version"] == "frames/3"
    assert set(diagnostics) >= {
        "published_region_count",
        "structural_context_count",
        "suppressed_regions",
        "suppressed_region_records",
    }


# Regressão da #239: a observação de cada frame era montada com os valores
# fixos de um fixture de teste, então dois frames diferentes saíam com o mesmo
# timestamp, a mesma posição na sequência e um artifact que não existia.
def test_each_frame_carries_its_own_recorded_provenance(tmp_path: Path) -> None:
    """Dois frames de um run têm timestamp, índice e artifact próprios."""
    frames_dir = tmp_path / "frames-in"
    _write_frame(frames_dir, frame_id="synthetic-000", sequence_index=3)
    _write_frame(frames_dir, frame_id="synthetic-001", sequence_index=9)

    out_dir = run_validation(
        ValidationOptions(frames_dir=frames_dir, results_dir=tmp_path / "out", sequence_masks=None),
        ports_factory=_fake_ports,
    )

    sources = {
        frame_id: json.loads((out_dir / "frames" / frame_id / "observation.json").read_text())["source"]
        for frame_id in ("synthetic-000", "synthetic-001")
    }
    first, second = sources["synthetic-000"], sources["synthetic-001"]
    assert first["sequence_index"] == 3 and second["sequence_index"] == 9
    assert first["timestamp"] != second["timestamp"]
    assert first["frame_id"] == "camera_optical_frame"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    recorded = {frame["frame_id"]: frame["input"]["provenance"] for frame in manifest["frames"]}
    assert recorded["synthetic-001"]["timestamp_ns"] == 1_900_000_000
    assert recorded["synthetic-000"]["artifact_uri"] == (frames_dir / "synthetic-000.png").resolve().as_uri()


# Um frame sem registro de proveniência não tem de onde tirar timestamp; o run
# precisa parar antes de carregar qualquer modelo, em vez de inventar valores.
def test_a_frame_without_provenance_stops_the_run_before_loading_models(tmp_path: Path) -> None:
    """Sem registro de proveniência, o run falha sem chamar a factory de ports."""
    frames_dir = tmp_path / "frames-in"
    _write_frame(frames_dir)
    (frames_dir / f"{_FRAME_ID}.json").unlink()

    def _forbidden_ports(config: ModuleConfig, lifecycle: ModelLifecycleManager) -> PerceptionPorts:
        raise AssertionError("ports must not be built for a frame without provenance")

    with pytest.raises(SystemExit, match="Re-extract"):
        run_validation(
            ValidationOptions(frames_dir=frames_dir, results_dir=tmp_path / "out", sequence_masks=None),
            ports_factory=_forbidden_ports,
        )


# Um discoverer que altera os pixels que recebe: simula exatamente o
# pré-processamento destrutivo que a garantia de imutabilidade existe para
# expor.
class _DestructiveRegionDiscoverer(FakeRegionDiscoverer):
    """Discoverer fake que zera a primeira linha da imagem antes de descobrir."""

    # Escreve sobre o buffer recebido e delega a descoberta ao fake original.
    def discover(self, image: ImagePayload, config: RegionDiscoveryConfig) -> tuple[LocalRegionProposal, ...]:
        """Zera a primeira linha dos pixels recebidos e descobre normalmente."""
        image.pixels[0, :, :] = 0
        return super().discover(image, config)


# Regressão da #240: o validador embrulhava o mesmo array como pixels de origem
# e como entrada do pipeline, então a comparação era verdadeira por identidade.
# Com cópias independentes, qualquer divergência entre as duas aparece.
def test_frame_inputs_detect_a_pipeline_input_that_differs_from_the_source() -> None:
    """Uma entrada que diverge da cópia de origem torna a comparação falsa."""
    source = np.full((4, 4, 3), 255, dtype=np.uint8)
    altered = source.copy()
    altered[0, :, :] = 0

    unchanged = FrameInputs(raw=ImagePayload(source.copy(), 4, 4), pipeline_input=ImagePayload(source, 4, 4))
    changed = FrameInputs(raw=ImagePayload(source.copy(), 4, 4), pipeline_input=ImagePayload(altered, 4, 4))

    assert unchanged.pipeline_input_is_raw() is True
    assert changed.pipeline_input_is_raw() is False


# Desde a #244 os pixels de um payload são somente-leitura: um adapter que
# tente alterar a imagem recebida falha na hora, em vez de corromper a entrada
# das views seguintes, e o frame de origem continua intacto.
def test_a_region_discoverer_cannot_alter_the_pixels_it_receives(tmp_path: Path) -> None:
    """Escrever nos pixels recebidos levanta erro e o frame de origem fica intacto."""
    frames_dir = tmp_path / "frames-in"
    source_pixels = _write_frame(frames_dir)

    def _destructive_ports(config: ModuleConfig, lifecycle: ModelLifecycleManager) -> PerceptionPorts:
        return dataclasses.replace(default_ports(), region_discoverer=_DestructiveRegionDiscoverer())

    with pytest.raises(ValueError, match="read-only"):
        run_validation(
            ValidationOptions(frames_dir=frames_dir, results_dir=tmp_path / "out", sequence_masks=None),
            ports_factory=_destructive_ports,
        )
    assert np.array_equal(np.array(Image.open(frames_dir / f"{_FRAME_ID}.png").convert("RGB")), source_pixels)


# Um caminho de geometria digitado errado não pode virar "rodar sem geometria":
# o run inteiro sairia sem exclusão do rig e da vinheta, sem nada no manifest.
def test_a_missing_sequence_masks_path_stops_the_run_before_loading_models(tmp_path: Path) -> None:
    """Geometria pedida e inexistente interrompe o run antes da factory de ports."""
    frames_dir = tmp_path / "frames-in"
    _write_frame(frames_dir)

    def _forbidden_ports(config: ModuleConfig, lifecycle: ModelLifecycleManager) -> PerceptionPorts:
        raise AssertionError("ports must not be built when the declared geometry is missing")

    with pytest.raises(SystemExit, match="sequence masks file not found"):
        run_validation(
            ValidationOptions(
                frames_dir=frames_dir,
                results_dir=tmp_path / "out",
                sequence_masks=tmp_path / "sequence-maks" / "corridor-02.json",
            ),
            ports_factory=_forbidden_ports,
        )


# O summary contém emoji e acentos. Sem encoding explícito, um locale ASCII
# levantava UnicodeEncodeError depois de todo o trabalho do run.
def test_the_run_completes_under_an_ascii_locale(tmp_path: Path) -> None:
    """Com locale C e sem modo UTF-8, o run escreve manifest e summary."""
    frames_dir = tmp_path / "frames-in"
    _write_frame(frames_dir)
    script = f"""
import sys
from pathlib import Path
sys.path[:0] = [{str(_THIS_DIR)!r}, {str(_THIS_DIR.parent / "tests")!r}]
from validate_reference_pipeline import ValidationOptions, run_validation
from fixtures_ports import default_ports
run_validation(
    ValidationOptions(frames_dir=Path({str(frames_dir)!r}), results_dir=Path({str(tmp_path / "out")!r}), sequence_masks=None),
    ports_factory=lambda config, lifecycle: default_ports(),
)
"""
    environment = {
        **os.environ,
        "LC_ALL": "C",
        "LANG": "C",
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script], env=environment, capture_output=True, text=True, check=False
    )

    assert completed.returncode == 0, completed.stderr
    (summary,) = (tmp_path / "out" / "samples").glob("*/summary.md")
    assert "✅" in summary.read_text(encoding="utf-8")


# Sem GPU, nenhum estágio mede VRAM: manifest e summary precisam dizer que não
# houve medida, em vez de apresentar zero ou o RSS do host como VRAM. Sem
# backend de grounding toda região sai ``grounding_unavailable``: a trilha
# registra isso, mas nenhum overlay semântico é escrito — antes, todo run sem
# grounder gravava um semantic-overlay.png idêntico a raw.png.
def test_memory_and_grounding_artifacts_report_only_what_was_measured(tmp_path: Path) -> None:
    """Sem medida de GPU e sem grounding aceito, nada é apresentado como se tivesse ocorrido."""
    frame_dir = _run(tmp_path)
    (run_dir,) = (tmp_path / "out" / "samples").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    diagnostics = json.loads((frame_dir / "diagnostics.json").read_text())

    assert manifest["frames"][0]["peak_vram_bytes"] is None
    assert "Pico de VRAM: não medido" in summary
    trail = json.loads((frame_dir / "DEBUG" / f"{_FRAME_ID}-grounding.json").read_text())
    assert {region["semantic_grounding_status"] for region in trail["regions"]} == {"grounding_unavailable"}
    assert not (frame_dir / "semantic-overlay.png").exists()
    assert diagnostics["semantic_overlay_written"] is False


# Regressão da #240: com grounding tentado e nenhuma máscara aceita, o overlay
# semântico saía idêntico a raw.png e era registrado como artifact de grounding.
def test_failed_grounding_writes_its_trail_but_no_semantic_overlay(tmp_path: Path) -> None:
    """Grounding só com falhas grava a trilha de debug e nenhum overlay."""
    payload = payload_with_blobs(width=64, height=64, blobs=((4, 4, 20, 20, (200, 30, 30)),))
    result = run_canonical_pipeline(
        image_observation(width=64, height=64), payload, ModuleConfig(), default_ports()
    )
    region = result.observation.regions[0]
    primary = primary_label_claim(region)
    assert primary is not None
    failed = SemanticGrounding(
        prediction=GroundingPrediction(region_id=region.region_id, concept=primary.value, reason="no detection"),
        status=GroundingStatus.FAILED,
        diagnostics={"reason": "no detection"},
    )
    observation = dataclasses.replace(
        result.observation,
        regions=tuple(
            dataclasses.replace(item, grounding=failed if item is region else None)
            for item in result.observation.regions
        ),
    )
    frame_dir = tmp_path / "frame"

    written = write_frame_artifacts(
        frame_dir,
        inputs=FrameInputs(raw=ImagePayload(payload.pixels.copy(), 64, 64), pipeline_input=payload),
        result=dataclasses.replace(result, observation=observation),
        diagnostics=diagnose_observation(observation, discovered_proposals=len(result.proposals)),
    )

    assert "semantic_overlay" not in written
    assert not (frame_dir / "semantic-overlay.png").exists()
    trail = json.loads((frame_dir / "DEBUG" / "frame-grounding.json").read_text())
    assert trail["regions"] == [{"reason": "no detection"}]
