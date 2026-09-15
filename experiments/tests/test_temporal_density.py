"""Testes da análise de densidade temporal de runs de percepção visual."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visual_perception_experiments.temporal_density import (
    FrameObservation,
    aggregate_totals,
    build_timeline,
    concept_persistence,
    entity_by_region_id,
    interesting_frames,
    published_concept_of,
    restrict_to_span,
    shared_frame_drift,
    structural_concept_counts,
)


# Monta uma claim com o mínimo que a leitura precisa. Existe para que cada teste
# declare apenas o que ele está exercitando, e não uma observação inteira.
def claim(value: str, *, role: str | None, category: str | None = None, stage: str = "region_semantics"):
    """Retorna uma claim serializada como o pipeline a escreve."""
    return {
        "kind": "label",
        "value": value,
        "role": role,
        "category": category,
        "region_kind": "thing",
        "confidence": {"value": 0.9, "source": "vlm"},
        "provenance": {"stage": stage, "producer": "test"},
    }


# Monta uma região publicada com as claims informadas.
def region(region_id: str, claims: list[dict], *, box: tuple[float, float, float, float] = (0, 0, 10, 10)):
    """Retorna uma região serializada como o pipeline a escreve."""
    x_min, y_min, x_max, y_max = box
    return {
        "region_id": region_id,
        "box": {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max},
        "claims": claims,
    }


def test_reconciled_claim_vence_a_primaria() -> None:
    payload = region(
        "region-a",
        [
            claim("smooth surface", role="primary", category="floor"),
            claim("wall", role="primary", category="building interior"),
            claim("wall", role="reconciled", category="building interior", stage="intra_frame_reconciliation"),
        ],
    )
    resolved = published_concept_of(payload)
    assert resolved.concept == "wall"
    assert resolved.category == "building interior"
    assert resolved.competing == ("smooth surface",)
    assert resolved.ambiguous is True


def test_primaria_unica_nao_e_ambigua() -> None:
    resolved = published_concept_of(region("region-b", [claim("wooden pallet", role="primary")]))
    assert resolved.concept == "wooden pallet"
    assert resolved.competing == ()
    assert resolved.ambiguous is False


def test_primarias_repetidas_nao_contam_como_divergencia() -> None:
    payload = region(
        "region-c",
        [claim("broken tile", role="primary"), claim("broken tile", role="primary")],
    )
    assert published_concept_of(payload).ambiguous is False


def test_conceito_canonico_da_publicacao_e_preservado() -> None:
    payload = region(
        "region-d",
        [
            claim("broken tile", role="primary", category="flooring"),
            claim("tile", role=None, stage="contextual_publication"),
        ],
    )
    assert published_concept_of(payload).canonical_concept == "tile"


def test_area_vem_da_caixa() -> None:
    resolved = published_concept_of(region("region-e", [claim("door", role="primary")], box=(10, 20, 40, 60)))
    assert resolved.area_px == pytest.approx(30 * 40)


def test_regiao_sem_claim_utilizavel_e_rejeitada() -> None:
    with pytest.raises(ValueError, match="no primary or reconciled claim"):
        published_concept_of(region("region-f", [claim("weathered", role=None)]))


def test_entidade_por_regiao_inverte_as_hipoteses() -> None:
    observation = {
        "entity_hypotheses": [
            {"entity_id": "entity-1", "member_region_ids": ["region-a", "region-b"]},
            {"entity_id": "entity-2", "member_region_ids": ["region-c"]},
        ]
    }
    assert entity_by_region_id(observation) == {
        "region-a": "entity-1",
        "region-b": "entity-1",
        "region-c": "entity-2",
    }


def test_contexto_estrutural_e_contado_por_conceito() -> None:
    observation = {
        "structural_context": [
            region("region-a", [claim("wall", role="primary")]),
            region("region-b", [claim("wall", role="primary")]),
            region("region-c", [claim("floor", role="primary")]),
        ]
    }
    assert structural_concept_counts(observation) == (("wall", 2), ("floor", 1))


# Constrói uma observação de frame já resolvida, para os testes que operam sobre
# a sequência e não sobre a leitura de arquivos.
def observation_at(
    index: int, time_s: float, concepts: list[str], *, published: int | None = None
) -> FrameObservation:
    """Retorna uma ``FrameObservation`` sintética posicionada no tempo."""
    resolved = tuple(
        published_concept_of(region(f"region-{index}-{position}", [claim(concept, role="primary")]))
        for position, concept in enumerate(concepts)
    )
    return FrameObservation(
        frame_id=f"corridor-02-{index:05d}",
        sequence_index=index,
        relative_time_s=time_s,
        proposal_count=10,
        region_count=5,
        published_region_count=len(concepts) if published is None else published,
        structural_context_count=4,
        published=resolved,
        structural_concepts=(("wall", 4),),
        entity_group_count=1,
        semantic_confidence_degenerate=True,
        latency_s=100.0,
    )


def test_persistencia_mede_primeiro_ultimo_e_continuidade() -> None:
    timeline = (
        observation_at(0, 0.0, ["wooden pallet"]),
        observation_at(1, 1.0, []),
        observation_at(2, 2.0, ["wooden pallet", "wooden pallet"]),
        observation_at(3, 3.0, ["door"]),
    )
    by_concept = {item.concept: item for item in concept_persistence(timeline)}

    pallet = by_concept["wooden pallet"]
    assert (pallet.first_seen_s, pallet.last_seen_s) == (0.0, 2.0)
    assert pallet.frame_count == 2
    assert pallet.span_frames == 3
    assert pallet.persistence == pytest.approx(2 / 3)
    assert pallet.region_count == 3
    assert pallet.frame_ids == ("corridor-02-00000", "corridor-02-00002")

    door = by_concept["door"]
    assert door.frame_count == door.span_frames == 1
    assert door.persistence == pytest.approx(1.0)


def test_totais_medem_a_maior_sequencia_cega() -> None:
    timeline = (
        observation_at(0, 0.0, ["door"]),
        observation_at(1, 1.0, []),
        observation_at(2, 2.0, []),
        observation_at(3, 3.0, []),
        observation_at(4, 4.0, ["door"]),
        observation_at(5, 5.0, []),
    )
    totals = aggregate_totals(timeline)
    assert totals.frame_count == 6
    assert totals.empty_frames == 4
    assert totals.longest_empty_run == 3
    assert totals.published_total == 2
    assert totals.concept_counts == (("door", 2),)


def test_totais_rejeitam_sequencia_vazia() -> None:
    with pytest.raises(ValueError, match="empty timeline"):
        aggregate_totals(())


def test_recorte_temporal_inclui_as_pontas() -> None:
    timeline = tuple(observation_at(index, float(index), ["door"]) for index in range(6))
    recorte = restrict_to_span(timeline, 1.0, 4.0)
    assert [frame.relative_time_s for frame in recorte] == [1.0, 2.0, 3.0, 4.0]


def test_recorte_tolera_o_desvio_do_header_em_relacao_ao_alvo() -> None:
    timeline = (observation_at(0, -0.014, ["door"]), observation_at(1, 15.019, ["door"]))
    assert len(restrict_to_span(timeline, 0.0, 15.0)) == 2


def test_recorte_rejeita_intervalo_invertido() -> None:
    with pytest.raises(ValueError, match="must not precede"):
        restrict_to_span((observation_at(0, 0.0, []),), 5.0, 1.0)


def test_desvio_compara_apenas_frames_compartilhados() -> None:
    baseline = (observation_at(0, 0.0, ["door"]), observation_at(2, 2.0, ["wooden pallet"]))
    candidate = (
        observation_at(0, 0.0, ["door"]),
        observation_at(1, 1.0, ["window"]),
        observation_at(2, 2.0, ["broken tile"]),
    )
    rows = shared_frame_drift(baseline, candidate)
    assert [row[0] for row in rows] == ["corridor-02-00000", "corridor-02-00002"]
    assert rows[0][3] == () and rows[0][4] == ()
    assert rows[1][3] == ("wooden pallet",)
    assert rows[1][4] == ("broken tile",)


def test_frames_de_interesse_cobrem_os_motivos_declarados() -> None:
    timeline = (
        observation_at(0, 0.0, ["window"]),
        observation_at(1, 1.0, ["broken tile", "debris"]),
        observation_at(2, 2.0, ["door"]),
    )
    motivos = dict(interesting_frames(timeline))
    assert motivos["primeiro frame"].frame_id == "corridor-02-00000"
    assert motivos["último frame"].frame_id == "corridor-02-00002"
    assert motivos["mais evidências publicadas"].frame_id == "corridor-02-00001"
    assert motivos["mais dano/detrito"].frame_id == "corridor-02-00001"
    assert motivos["door na maior área"].frame_id == "corridor-02-00002"


def test_frames_de_interesse_rejeitam_sequencia_vazia() -> None:
    with pytest.raises(ValueError, match="empty timeline"):
        interesting_frames(())


# Escreve em disco a estrutura mínima de janela e de run que ``build_timeline``
# consome, para exercitar a junção das duas fontes sem depender de um run real.
def write_run(root: Path, *, frames: list[tuple[int, int, list[str]]]) -> tuple[Path, Path]:
    """Cria uma janela e um run sintéticos e devolve seus caminhos."""
    start_ns = 1_000_000_000_000
    window = {
        "camera_topic": "/camera_1/image_raw",
        "start_header_ns": start_ns,
        "end_header_ns": start_ns + 10_000_000_000,
        "play_offset_s": 1.0,
        "play_duration_s": 13.0,
        "lead_s": 3.0,
        "frame_id_prefix": "corridor-02",
        "keyframes": [
            {
                "sequence_index": index,
                "header_timestamp_ns": start_ns + offset_ns,
                "bag_timestamp_ns": start_ns + offset_ns,
            }
            for index, offset_ns, _ in frames
        ],
    }
    window_path = root / "window.json"
    window_path.write_text(json.dumps(window), encoding="utf-8")

    run = root / "run"
    reports = []
    for index, _, concepts in frames:
        frame_id = f"corridor-02-{index:05d}"
        frame_dir = run / "frames" / frame_id
        frame_dir.mkdir(parents=True)
        observation = {
            "regions": [
                region(f"region-{position}", [claim(concept, role="primary")])
                for position, concept in enumerate(concepts)
            ],
            "structural_context": [region("region-wall", [claim("wall", role="primary")])],
            "entity_hypotheses": [],
        }
        (frame_dir / "observation.json").write_text(json.dumps(observation), encoding="utf-8")
        (frame_dir / "diagnostics.json").write_text(
            json.dumps({"semantic_confidence": {"degenerate": True}}), encoding="utf-8"
        )
        reports.append(
            {
                "frame_id": frame_id,
                "failed": False,
                "proposal_count": 12,
                "canonical_region_count": len(concepts) + 1,
                "published_region_count": len(concepts),
                "structural_context_count": 1,
                "latency_s": 200.0,
            }
        )
    (run / "manifest.json").write_text(json.dumps({"frames": reports}), encoding="utf-8")
    return window_path, run


def test_timeline_junta_a_janela_com_o_run_em_ordem_cronologica(tmp_path: Path) -> None:
    window, run = write_run(
        tmp_path,
        frames=[
            (4282, 2_000_000_000, ["wooden pallet"]),
            (4234, 0, ["door"]),
            (4331, 4_000_000_000, []),
        ],
    )
    timeline = build_timeline(window, run)
    assert [frame.frame_id for frame in timeline] == [
        "corridor-02-04234",
        "corridor-02-04282",
        "corridor-02-04331",
    ]
    assert [frame.relative_time_s for frame in timeline] == [0.0, 2.0, 4.0]
    assert timeline[0].published[0].concept == "door"
    assert timeline[2].published == ()
    assert timeline[0].structural_concepts == (("wall", 1),)
    assert timeline[0].semantic_confidence_degenerate is True


def test_timeline_ignora_keyframe_sem_artifacts(tmp_path: Path) -> None:
    window, run = write_run(tmp_path, frames=[(4234, 0, ["door"]), (4282, 2_000_000_000, ["window"])])
    payload = json.loads(window.read_text(encoding="utf-8"))
    payload["keyframes"].append(
        {"sequence_index": 9999, "header_timestamp_ns": 1_000_003_000_000_000, "bag_timestamp_ns": 0}
    )
    window.write_text(json.dumps(payload), encoding="utf-8")
    assert [frame.sequence_index for frame in build_timeline(window, run)] == [4234, 4282]


def test_timeline_sem_nenhum_frame_resolvido_e_rejeitada(tmp_path: Path) -> None:
    window, run = write_run(tmp_path, frames=[(4234, 0, ["door"])])
    payload = json.loads(window.read_text(encoding="utf-8"))
    payload["keyframes"] = [
        {"sequence_index": 1, "header_timestamp_ns": 1_000_000_000_000, "bag_timestamp_ns": 0}
    ]
    window.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="has perception artifacts"):
        build_timeline(window, run)
