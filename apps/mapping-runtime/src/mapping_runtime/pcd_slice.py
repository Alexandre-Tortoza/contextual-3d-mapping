"""Importação limitada de mapas PCD para inspeção no map-explorer."""

from __future__ import annotations

import json
import struct
from hashlib import sha256
from math import ceil, isfinite
from pathlib import Path
from typing import BinaryIO


# Lê o cabeçalho PCD sem consumir o payload binário. Existe para manter o
# formato externo contido no composition root e produzir erros acionáveis.
def _read_header(stream: BinaryIO) -> tuple[dict[str, list[str]], int]:
    """Lê e valida o cabeçalho de um arquivo PCD.

    Argumentos:
        stream: arquivo binário posicionado no início.
    Retorna:
        campos normalizados do cabeçalho e offset inicial dos pontos.
    """
    header: dict[str, list[str]] = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD header ended before DATA.")
        try:
            text = line.decode("ascii").strip()
        except UnicodeDecodeError as error:
            raise ValueError("PCD header must be ASCII.") from error
        if not text or text.startswith("#"):
            continue
        key, *values = text.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            break
    if header["DATA"] != ["binary"]:
        raise ValueError("only DATA binary PCD files are supported.")
    return header, stream.tell()


# Mapeia os escalares aceitos pelo PCD para formatos little-endian do Python.
# A fronteira é estreita porque o slice precisa apenas de x, y, z e intensidade.
def _scalar_format(field_type: str, size: int) -> str:
    """Retorna o formato struct de um campo escalar PCD.

    Argumentos:
        field_type: código PCD ``F``, ``I`` ou ``U``.
        size: largura do campo em bytes.
    Retorna:
        código de formato compatível com ``struct.unpack_from``.
    """
    formats = {
        ("F", 4): "f",
        ("F", 8): "d",
        ("I", 1): "b",
        ("I", 2): "h",
        ("I", 4): "i",
        ("U", 1): "B",
        ("U", 2): "H",
        ("U", 4): "I",
    }
    try:
        return formats[(field_type, size)]
    except KeyError as error:
        raise ValueError(f"unsupported PCD scalar: {field_type}{size}.") from error


# Calcula offsets dos campos descritos no cabeçalho para ler registros sem
# depender de PCL ou de uma biblioteca Python externa.
def _field_layout(header: dict[str, list[str]]) -> tuple[dict[str, tuple[int, str]], int]:
    """Converte o schema PCD em offsets e formatos escalares.

    Argumentos:
        header: cabeçalho produzido por ``_read_header``.
    Retorna:
        layout por nome de campo e tamanho total de cada ponto.
    """
    try:
        names = header["FIELDS"]
        sizes = [int(value) for value in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(names))]
    except (KeyError, ValueError) as error:
        raise ValueError("invalid PCD field declaration.") from error
    if not (len(names) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD FIELDS, SIZE, TYPE and COUNT must have equal lengths.")
    layout: dict[str, tuple[int, str]] = {}
    offset = 0
    for name, size, field_type, count in zip(names, sizes, types, counts, strict=True):
        if size <= 0 or count <= 0:
            raise ValueError("PCD SIZE and COUNT values must be positive.")
        if count == 1:
            layout[name] = (offset, _scalar_format(field_type, size))
        offset += size * count
    if not {"x", "y", "z"}.issubset(layout):
        raise ValueError("PCD must contain scalar x, y and z fields.")
    return layout, offset


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
    with source.open("rb") as stream:
        header, data_offset = _read_header(stream)
        layout, point_step = _field_layout(header)
        try:
            point_count = int(header["POINTS"][0])
        except (KeyError, IndexError, ValueError) as error:
            raise ValueError("PCD POINTS must be an integer.") from error
        payload = stream.read()
    expected_size = point_count * point_step
    if point_count < 0 or len(payload) < expected_size:
        raise ValueError("PCD binary payload is truncated.")
    stride = max(1, ceil(point_count / max_points))
    sampled: list[tuple[int, tuple[float, float, float], float | None]] = []
    for index in range(0, point_count, stride):
        base = index * point_step
        coordinates = tuple(
            float(struct.unpack_from("<" + layout[axis][1], payload, base + layout[axis][0])[0])
            for axis in ("x", "y", "z")
        )
        if not all(isfinite(value) for value in coordinates):
            continue
        intensity = None
        if "intensity" in layout:
            intensity = float(
                struct.unpack_from("<" + layout["intensity"][1], payload, base + layout["intensity"][0])[0]
            )
        sampled.append((index, coordinates, intensity))
    if not sampled:
        raise ValueError("PCD does not contain finite sampled points.")
    heights = [coordinates[2] for _, coordinates, _ in sampled]
    minimum_height, maximum_height = min(heights), max(heights)
    points = [
        {
            "geometry_id": f"{map_id}:pcd:{index}",
            "coordinates_m": coordinates,
            "source_coordinates_m": coordinates,
            "intensity": intensity,
            "display_color_rgb": _height_color(coordinates[2], minimum_height, maximum_height),
        }
        for index, coordinates, intensity in sampled
    ]
    payload_record = {
        "schema_version": 1,
        "artifact_type": "geometric_pcd_slice",
        "map_id": map_id,
        "map_frame": map_frame,
        "source": {
            "uri": str(source),
            "media_type": "application/vnd.pointclouds.pcd",
            "sha256": sha256(source.read_bytes()).hexdigest(),
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
