"""Exportação versionada de um slice RGB–LiDAR para aplicações consumidoras."""

from __future__ import annotations

import json
from pathlib import Path

from geometric_map import GeometryPoint
from sensor_association import PointVisualAssociation


# Converte um ponto e sua evidência associada para um artifact simples e
# auditável. Existe no composition root porque não muda ownership de nenhum módulo.
def _record(point: GeometryPoint, association: PointVisualAssociation | None) -> dict[str, object]:
    """Serializa um ponto do mapa e sua associação visual opcional."""
    record: dict[str, object] = {
        "geometry_id": point.reference.geometry_id,
        "coordinates_m": point.coordinates_m,
        "source_coordinates_m": point.source_coordinates_m,
        "lidar_observation_id": point.source_observation.observation_id,
        "lidar_timestamp_ns": point.source_observation.timestamp.nanoseconds,
    }
    if association is not None:
        record["association"] = {
            "status": association.status.value,
            "rgb_observation_id": association.rgb_observation.observation_id,
            "rgb_timestamp_ns": association.rgb_observation.timestamp.nanoseconds,
            "pixel": association.pixel,
            "color_rgb": association.color_rgb,
            "region_id": association.region_id,
            "label": association.label,
            "feature_reference": association.feature_reference,
            "calibration_id": association.calibration.calibration_id,
            "calibration_artifact": association.calibration.artifact.uri,
        }
    return record


# Produz o formato inicial de intercâmbio do map-explorer. A escrita atômica
# impede que o viewer abra um JSON parcialmente atualizado durante uma execução.
def export_associated_slice(
    destination: Path,
    map_id: str,
    map_frame: str,
    points: tuple[GeometryPoint, ...],
    associations: tuple[PointVisualAssociation, ...],
) -> Path:
    """Exporta um slice associado versionado para o caminho solicitado.

    Argumentos:
        destination: arquivo JSON final do artifact.
        map_id: identidade do mapa proprietário.
        map_frame: frame global das coordenadas persistidas.
        points: pontos geométricos a exportar.
        associations: associações visuais ancoradas nesses pontos.
    Retorna:
        caminho do artifact persistido.
    """
    by_geometry = {item.geometry.geometry_id: item for item in associations}
    payload = {
        "schema_version": 1,
        "map_id": map_id,
        "map_frame": map_frame,
        "points": [_record(point, by_geometry.get(point.reference.geometry_id)) for point in points],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination
