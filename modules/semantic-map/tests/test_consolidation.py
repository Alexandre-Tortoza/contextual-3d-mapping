"""Regressões da consolidação semântica entre runs publicadas."""

from __future__ import annotations

import pytest

from semantic_map import PublishedContextRun, consolidate_context_runs, geometry_fingerprint


# Cria um artifact contextual pequeno, mas completo o bastante para atravessar
# a fronteira pública sem depender de ROS, imagens reais ou inferência visual.
def _payload(label: str, *, confidence: float = 0.5, coordinate: float = 0.0) -> dict:
    """Materializa uma run com um único ponto e uma evidência semântica."""
    return {
        "schema_version": 2,
        "artifact_type": "contextual_rgb_lidar_slice",
        "map_id": "shared-map",
        "map_frame": "map",
        "points": [{
            "geometry_id": "shared-map:pcd:0",
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


# Um mesmo map_id não basta para a fusão: uma coordenada diferente sob a mesma
# identidade de ponto denuncia que as runs não compartilham a mesma geometria.
def test_consolidation_rejects_different_geometry() -> None:
    """Recusa runs que só parecem pertencer ao mesmo mapa."""
    first = _payload("door")
    second = _payload("door", coordinate=1.0)
    assert geometry_fingerprint(first) != geometry_fingerprint(second)
    with pytest.raises(ValueError, match="same geometry"):
        consolidate_context_runs((_run("first", first), _run("second", second)))
