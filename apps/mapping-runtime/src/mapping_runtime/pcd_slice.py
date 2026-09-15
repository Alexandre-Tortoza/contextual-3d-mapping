"""Importação limitada de mapas PCD para inspeção no map-explorer."""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

from geometric_map import read_pcd_geometry

from contextual_mapping_contracts import FrameId, MapId


# Converte altura em uma cor apenas de visualização. Ela não é publicada como
# associação RGB e, portanto, não cria evidência semântica falsa no artifact.
def _height_color(height: float, minimum: float, maximum: float) -> tuple[int, int, int]:
    """Produz uma cor azul–verde–amarela para a altura de um ponto.

    Argumentos:
        height: coordenada vertical do ponto.
        minimum: menor altura da amostra.
        maximum: maior altura da amostra.
    Retorna:
        tripla RGB usada exclusivamente pelo viewer.
    """
    ratio = 0.5 if maximum == minimum else (height - minimum) / (maximum - minimum)
    red = round(35 + 220 * ratio)
    green = round(100 + 140 * (1.0 - abs(2.0 * ratio - 1.0)))
    blue = round(230 - 190 * ratio)
    return red, green, blue


# Amostra deterministicamente um PCD binário e o exporta no schema consumido
# pelo viewer. Existe para inspecionar trechos FAST-LIO sem carregar PCL no app.
# Décimo de milímetro. O PCD guarda os pontos em float32 e o LiDAR resolve
# centímetros, então as casas decimais além desta são ruído de representação:
# elas custam mais bytes por ponto do que a geometria inteira do slice.
_COORDINATE_DECIMALS = 4


def geometry_point_records(
    map_id: str, sampled: list[tuple[int, tuple[float, ...], float | None]]
) -> list[dict]:
    """Converte pontos ``(índice de origem, xyz, intensidade)`` no schema do viewer.

    Argumentos:
        map_id: identidade do mapa, prefixo estável de ``geometry_id``.
        sampled: pontos já selecionados, na ordem de exportação.
    Retorna:
        registros com identidade, coordenadas arredondadas, intensidade e cor por altura.
    """
    heights = [coordinates[2] for _, coordinates, _ in sampled]
    minimum_height, maximum_height = min(heights), max(heights)
    return [
        {
            "geometry_id": f"{map_id}:pcd:{index}",
            "coordinates_m": [round(value, _COORDINATE_DECIMALS) for value in coordinates],
            "intensity": intensity,
            "display_color_rgb": _height_color(coordinates[2], minimum_height, maximum_height),
        }
        for index, coordinates, intensity in sampled
    ]


# Amostra um PCD inteiro com passo fixo pelo índice de origem e grava o slice.
def export_pcd_slice(
    source: Path,
    destination: Path,
    *,
    map_id: str,
    map_frame: str,
    max_points: int = 25_000,
) -> Path:
    """Exporta uma amostra geométrica de um PCD para o map-explorer.

    Argumentos:
        source: mapa PCD binário produzido pelo FAST-LIO.
        destination: artifact JSON de destino.
        map_id: identidade estável do mapa.
        map_frame: frame global das coordenadas.
        max_points: limite determinístico de pontos no viewer.
    Retorna:
        caminho do artifact JSON criado atomicamente.
    """
    if not map_id.strip() or not map_frame.strip():
        raise ValueError("map_id and map_frame must not be empty.")
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise TypeError("max_points must be an integer.")
    if max_points <= 0:
        raise ValueError("max_points must be positive.")
    cloud = read_pcd_geometry(source, map_id=MapId(map_id), frame_id=FrameId(map_frame))
    point_count, data_offset = cloud.point_count, cloud.data_offset
    stride = max(1, ceil(point_count / max_points))
    sampled = [
        (int(index), tuple(float(value) for value in coordinates),
         None if cloud.intensities is None else float(cloud.intensities[position]))
        for position, (index, coordinates) in enumerate(zip(cloud.source_indices, cloud.coordinates_m, strict=True))
        if index % stride == 0
    ]
    if not sampled:
        raise ValueError("PCD does not contain finite sampled points.")
    points = geometry_point_records(map_id, sampled)
    payload_record = {
        "schema_version": 1,
        "artifact_type": "geometric_pcd_slice",
        "map_id": map_id,
        "map_frame": map_frame,
        "source": {
            "uri": str(source),
            "media_type": "application/vnd.pointclouds.pcd",
            "sha256": cloud.source.digest,
            "point_count": point_count,
            "data_offset": data_offset,
            "sampling_stride": stride,
        },
        "points": points,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload_record, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
