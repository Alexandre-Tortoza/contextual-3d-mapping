"""Regressões da consolidação semântica entre runs publicadas."""

from __future__ import annotations

import pytest

from semantic_map import PublishedContextRun, consolidate_context_runs, geometry_fingerprint


# Cria um artifact contextual pequeno, mas completo o bastante para atravessar
# a fronteira pública sem depender de ROS, imagens reais ou inferência visual.
def _payload(
    label: str, *, confidence: float = 0.5, coordinate: float = 0.0, index: int = 0, source_sha: str = "a" * 64,
    observation_debug_manifest: dict | None = None, composition_uri: str | None = None,
    pipeline_backends: dict | None = None,
) -> dict:
    """Materializa uma run com um único ponto e uma evidência semântica."""
    observation = {
        "observation_id": "camera-0", "timestamp_ns": 10, "width": 4, "height": 4,
        "raw_image_uri": "assets/raw.png", "overlay_image_uri": "assets/overlay.png",
    }
    if observation_debug_manifest is not None:
        observation["debug_manifest"] = observation_debug_manifest
    payload = {
        "schema_version": 3,
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
        "observations": [observation],
        "regions": [{
            "region_id": "region-0", "label": label,
            "confidence": {"value": confidence}, "visual_support": confidence,
            "region_quality": confidence,
        }],
    }
    if composition_uri is not None or pipeline_backends is not None:
        debug_manifest = payload.setdefault("debug_manifest", {})
        if composition_uri is not None:
            debug_manifest["composition"] = composition_uri
        if pipeline_backends is not None:
            debug_manifest["pipeline_backends"] = pipeline_backends
    return payload


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
    # O consolidado reusa a mesma forma de documento das runs de origem, então
    # acompanha o mesmo schema_version — não é um schema à parte com número
    # próprio que fica esquecido a cada bump da run individual.
    assert result.payload["schema_version"] == _payload("pallet")["schema_version"]


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


# map_id e map_frame são o contract estrutural que garante que runs fundidas
# compartilham a mesma origem geométrica; uma identidade vazia não deve
# atravessar a fronteira pública do módulo silenciosamente.
def test_published_run_rejects_empty_map_identity() -> None:
    """Recusa map_id ou map_frame vazio na construção da run publicada."""
    with pytest.raises(ValueError, match="map id"):
        _run("first", {**_payload("door"), "map_id": ""})
    with pytest.raises(ValueError, match="frame id"):
        _run("first", {**_payload("door"), "map_frame": " "})


# O fingerprint é a única defesa contra fundir origens diferentes; uma
# identidade vazia não pode produzir um hash válido silenciosamente.
def test_geometry_fingerprint_rejects_empty_map_identity() -> None:
    """Recusa map_id ou map_frame vazio ao calcular o fingerprint de origem."""
    with pytest.raises(ValueError, match="valid map_id"):
        geometry_fingerprint({**_payload("door"), "map_id": ""})
    with pytest.raises(ValueError, match="valid map_id"):
        geometry_fingerprint({**_payload("door"), "map_frame": " "})


# Antes desta mudança, ``debug_manifest`` (como ``debug_assets``) não passava
# por ``_asset_url``: os links de debug do mapa consolidado apontavam para
# caminhos relativos à run de origem em vez de ``/runs/<run_id>/...``, e
# quebravam. A reescrita precisa alcançar toda folha, inclusive dentro de
# listas (``discovery_tiles``, ``region_views``), e nunca colidir entre runs.
def test_consolidation_rewrites_debug_manifest_uris_per_run() -> None:
    """Reescreve debug_manifest de cada run para sua própria URL pública, sem colisão."""
    first = _payload(
        "door",
        observation_debug_manifest={
            "visual-perception": {
                "diagnostics": "assets/frame-diagnostics.json",
                "images": {"raw": "assets/frame-debug/raw.png"},
                "discovery_tiles": [{"id": "tile-0", "input": "assets/frame-debug/discovery/tile-0-input.png"}],
                "region_views": [{"region_id": "region-0", "tight_crop": "assets/frame-debug/regions/region-0/tight-crop.png"}],
            },
        },
    )
    second = _payload(
        "door",
        index=1,
        coordinate=1.0,
        observation_debug_manifest={"visual-perception": {"diagnostics": "assets/frame-diagnostics.json"}},
    )

    result = consolidate_context_runs((_run("first", first), _run("second", second)))

    by_run = {
        observation["source_run_id"]: observation["debug_manifest"]
        for observation in result.payload["observations"]
    }
    assert by_run["first"]["visual-perception"]["diagnostics"] == "/runs/first/assets/frame-diagnostics.json"
    assert by_run["first"]["visual-perception"]["images"]["raw"] == "/runs/first/assets/frame-debug/raw.png"
    assert by_run["first"]["visual-perception"]["discovery_tiles"][0]["input"] == (
        "/runs/first/assets/frame-debug/discovery/tile-0-input.png"
    )
    assert by_run["first"]["visual-perception"]["region_views"][0]["tight_crop"] == (
        "/runs/first/assets/frame-debug/regions/region-0/tight-crop.png"
    )
    assert by_run["second"]["visual-perception"]["diagnostics"] == "/runs/second/assets/frame-diagnostics.json"


# ``debug_manifest.composition`` de uma run individual é uma única string; o
# consolidado funde várias runs, então a mesma chave vira uma lista com uma
# entrada por run de origem que declarou composição.
def test_consolidation_turns_composition_into_a_list_per_run() -> None:
    """Duas runs com composição viram duas entradas rastreáveis no consolidado."""
    first = _payload("door", composition_uri="assets/composition.json")
    second = _payload("door", index=1, coordinate=1.0, composition_uri="assets/composition.json")

    result = consolidate_context_runs((_run("first", first), _run("second", second)))

    composition = result.payload["debug_manifest"]["composition"]
    assert {entry["run_id"] for entry in composition} == {"first", "second"}
    by_run = {entry["run_id"]: entry["uri"] for entry in composition}
    assert by_run["first"] == "/runs/first/assets/composition.json"
    assert by_run["second"] == "/runs/second/assets/composition.json"


# ``pipeline_backends`` não é uma URI, então não passa por _asset_url; mas
# precisa preservar de qual run veio cada configuração para uma comparação
# entre backends fazer sentido no consolidado.
def test_consolidation_preserves_pipeline_backends_per_run() -> None:
    """Cada run mantém sua própria configuração de backends, rastreável por run_id."""
    first = _payload("door", pipeline_backends={"region_discovery": {"backend": "sam3", "checkpoint": "facebook/sam3"}})
    second = _payload(
        "door", index=1, coordinate=1.0,
        pipeline_backends={"region_discovery": {"backend": "sam2", "checkpoint": "facebook/sam2"}},
    )

    result = consolidate_context_runs((_run("first", first), _run("second", second)))

    backends = result.payload["debug_manifest"]["pipeline_backends"]
    by_run = {entry["run_id"]: entry["backends"] for entry in backends}
    assert by_run["first"]["region_discovery"]["backend"] == "sam3"
    assert by_run["second"]["region_discovery"]["backend"] == "sam2"
