"""Regressões da consolidação semântica entre runs publicadas."""

from __future__ import annotations

import pytest

from semantic_map import PublishedContextRun, consolidate_context_runs, geometry_fingerprint


# Cria um artifact contextual pequeno, mas completo o bastante para atravessar
# a fronteira pública sem depender de ROS, imagens reais ou inferência visual.
def _payload(
    label: str, *, confidence: float = 0.5, coordinate: float = 0.0, index: int = 0, source_sha: str = "a" * 64,
) -> dict:
    """Materializa uma run com um único ponto e uma evidência semântica."""
    return {
        "schema_version": 2,
        "artifact_type": "contextual_rgb_lidar_slice",
        "map_id": "shared-map",
        "map_frame": "map",
        "source": {"sha256": source_sha, "point_count": 100},
        "points": [{
            "geometry_id": f"shared-map:pcd:{index}",
            "coordinates_m": [coordinate, 0.0, 0.0],
            "context": {
                "status": "associated", "observation_id": "camera-0", "region_id": "region-0",
                "label": label, "confidence": confidence, "pixel": [2, 3],
            },
        }],
        "observations": [{
            "observation_id": "camera-0", "timestamp_ns": 10, "width": 4, "height": 4,
            "raw_image_uri": "assets/raw.png", "overlay_image_uri": "assets/overlay.png",
        }],
        "regions": [{
            "region_id": "region-0", "label": label,
            "confidence": {"value": confidence}, "visual_support": confidence,
            "region_quality": confidence,
        }],
    }


# Encapsula os metadados que o publisher fornece ao módulo, mantendo os testes
# focados na capacidade e não no layout estático do map-explorer.
def _run(run_id: str, payload: dict) -> PublishedContextRun:
    """Cria uma run publicada para a consolidação."""
    return PublishedContextRun(run_id, run_id, f"/runs/{run_id}/context.json", run_id * 8, payload)


# Duas variantes textuais devem somar votos para a mesma família canônica, em
# vez de competir como se descrevessem objetos independentes.
def test_consolidation_groups_textual_variants_and_keeps_all_runs() -> None:
    """Agrupa pallet e wooden pallet em um resultado rastreável."""
    result = consolidate_context_runs((_run("first", _payload("pallet")), _run("second", _payload("wooden pallet"))))
    point = result.payload["points"][0]
    context = point["context"]
    assert context["label"] == "pallet"
    assert context["agreement"] == 1.0
    assert context["consolidation"]["resolution"] == "majority"
    assert {item["source_run_id"] for item in context["contributions"]} == {"first", "second"}
    assert result.payload["observations"][0]["raw_image_uri"].startswith("/runs/")


# Sem maioria, a escolha continua determinística pelo ranking de qualidade e
# informa explicitamente que o claim não representa consenso entre as runs.
def test_consolidation_uses_quality_tiebreak_without_majority() -> None:
    """Escolhe o claim mais confiável e o marca como desempate."""
    result = consolidate_context_runs((_run("first", _payload("door", confidence=0.2)), _run("second", _payload("window", confidence=0.9))))
    context = result.payload["points"][0]["context"]
    assert context["label"] == "window"
    assert context["agreement"] == 0.5
    assert context["consolidation"]["resolution"] == "quality_tiebreak"
    assert context["support_state"] == "weak"


# Runs de nuvens de origem diferentes não são fundidas, e dentro da mesma origem
# uma identidade de ponto com coordenadas diferentes denuncia artifacts incoerentes.
def test_consolidation_rejects_different_geometry() -> None:
    """Recusa origem divergente e coordenadas conflitantes para o mesmo ponto."""
    first = _payload("door")
    other_source = _payload("door", source_sha="b" * 64)
    assert geometry_fingerprint(first) != geometry_fingerprint(other_source)
    with pytest.raises(ValueError, match="same geometry source"):
        consolidate_context_runs((_run("first", first), _run("second", other_source)))
    with pytest.raises(ValueError, match="different coordinates"):
        consolidate_context_runs((_run("first", first), _run("second", _payload("door", coordinate=1.0))))


# Trechos carregam recortes diferentes da mesma nuvem: sem fundo, o mapa geral é a
# união dos pontos, e cada ponto só recebe votos das runs que o contêm.
def test_consolidation_unites_disjoint_segment_crops() -> None:
    """Funde runs com pontos disjuntos da mesma origem."""
    first = _payload("door", index=1)
    second = _payload("pallet", index=2, coordinate=5.0)
    assert geometry_fingerprint(first) == geometry_fingerprint(second)

    points = {point["geometry_id"]: point for point in consolidate_context_runs((_run("first", first), _run("second", second))).payload["points"]}

    assert set(points) == {"shared-map:pcd:1", "shared-map:pcd:2"}
    assert points["shared-map:pcd:1"]["context"]["label"] == "door"
    assert points["shared-map:pcd:2"]["context"]["consolidation"]["observed_run_count"] == 1


# Com o fundo global, a geometria do mapa geral é o fundo mais os pontos com contexto
# das runs; pontos sem contexto de um recorte não inflam o artifact.
def test_consolidation_uses_the_global_backdrop_as_geometry() -> None:
    """Mantém o fundo, acrescenta só pontos contextuais e recusa fundo de outra origem."""
    run = _payload("door", index=7, coordinate=3.0)
    run["points"].append({"geometry_id": "shared-map:pcd:8", "coordinates_m": [4.0, 0.0, 0.0], "context": {"status": "occluded"}})
    backdrop = {
        "artifact_type": "geometric_pcd_slice", "map_id": "shared-map", "map_frame": "map",
        "source": {"sha256": "a" * 64, "point_count": 100},
        "points": [{"geometry_id": f"shared-map:pcd:{index}", "coordinates_m": [float(index) * 10, 0.0, 0.0]} for index in (0, 5)],
    }

    result = consolidate_context_runs((_run("first", run),), backdrop=backdrop)

    ids = [point["geometry_id"] for point in result.payload["points"]]
    assert ids == ["shared-map:pcd:0", "shared-map:pcd:5", "shared-map:pcd:7"]
    assert result.payload["points"][2]["context"]["label"] == "door"
    assert result.payload["consolidation"]["backdrop_point_count"] == 2
    with pytest.raises(ValueError, match="backdrop"):
        consolidate_context_runs((_run("first", run),), backdrop={**backdrop, "source": {"sha256": "c" * 64, "point_count": 100}})
