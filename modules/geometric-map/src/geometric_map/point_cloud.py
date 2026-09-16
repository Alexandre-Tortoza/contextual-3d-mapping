"""Leitura de geometria PCD medida, independente da amostragem do viewer."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO

import numpy as np

from contextual_mapping_contracts import FrameId, MapId, SourceArtifactReference


# Lê o cabeçalho PCD sem consumir o payload binário. Existe para manter o
# formato externo contido no módulo de geometria e produzir erros acionáveis.
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


# Preserva a geometria medida completa e sua identidade, para que o cálculo de
# visibilidade não dependa dos pontos escolhidos para exibição no navegador.
@dataclass(frozen=True)
class PointCloudGeometry:
    """Nuvem medida em metros no frame do mapa, com índices da origem.

    Os arrays são copiados e protegidos contra escrita na fronteira pública.
    Índices referem-se aos registros originais, inclusive após remover NaNs.
    """

    coordinates_m: np.ndarray
    source_indices: np.ndarray
    map_id: MapId
    frame_id: FrameId
    source: SourceArtifactReference
    point_count: int
    data_offset: int = 0
    intensities: np.ndarray | None = None

    # Valida frames e arrays antes que uma nuvem seja usada como oclusor.
    def __post_init__(self) -> None:
        """Exige coordenadas finitas e índices únicos da geometria medida."""
        xyz = np.array(self.coordinates_m, dtype=np.float64, copy=True)
        if type(self.point_count) is not int or self.point_count <= 0:
            raise ValueError("A contagem da origem deve ser um inteiro positivo.")
        if not np.issubdtype(np.asarray(self.source_indices).dtype, np.integer):
            raise ValueError("Os índices da origem devem ser inteiros.")
        indices = np.array(self.source_indices, dtype=np.int64, copy=True)
        if xyz.ndim != 2 or xyz.shape[1:] != (3,) or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError("A nuvem deve conter coordenadas XYZ finitas, em metros.")
        if indices.shape != (len(xyz),) or len(np.unique(indices)) != len(indices):
            raise ValueError("Os índices da nuvem devem ser únicos e acompanhar as coordenadas.")
        if (indices < 0).any() or (indices >= self.point_count).any():
            raise ValueError("Índices fora da contagem de pontos da origem.")
        if not str(self.map_id).strip() or not str(self.frame_id).strip():
            raise ValueError("A nuvem exige map_id e frame_id.")
        for name, value in (("coordinates_m", xyz), ("source_indices", indices)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.intensities is not None:
            intensities = np.array(self.intensities, dtype=np.float64, copy=True)
            if intensities.shape != (len(xyz),):
                raise ValueError("Intensidades devem acompanhar as coordenadas.")
            intensities.setflags(write=False)
            object.__setattr__(self, "intensities", intensities)


# Seleciona a geometria medida em torno de um trecho: pontos a até ``radius_m`` de
# algum centro, amostrados de forma determinística até ``max_points``. Existe para
# que o artifact de um trecho carregue a vizinhança densa em vez do mapa global
# inteiro amostrado. Os índices de origem são preservados, então a identidade de
# cada ponto continua comparável entre trechos e com o mapa global.
def select_neighbourhood(
    cloud: PointCloudGeometry, centers_m: np.ndarray, *, centers_frame: FrameId, radius_m: float, max_points: int
) -> tuple[PointCloudGeometry, int]:
    """Retorna os pontos próximos aos centros e o passo de amostragem aplicado.

    Argumentos:
        cloud: nuvem medida completa, no frame do mapa.
        centers_m: centros ``(N, 3)`` no mesmo frame, em metros.
        centers_frame: frame em que ``centers_m`` foi expresso; deve coincidir com ``cloud.frame_id``.
        radius_m: raio máximo até o centro mais próximo.
        max_points: teto de pontos da seleção.
    Retorna:
        ``(nuvem selecionada, passo)``; o passo é 1 quando a seleção cabe no teto.
    Levanta:
        ValueError: se os frames divergirem, ou se centros, raio ou teto forem inválidos, ou
            nenhum ponto couber no raio.
    """
    if centers_frame != cloud.frame_id:
        raise ValueError(
            f"camera centers are expressed in frame {centers_frame!r}, "
            f"but the point cloud is in frame {cloud.frame_id!r}."
        )
    centers = np.asarray(centers_m, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1:] != (3,) or not len(centers) or not np.isfinite(centers).all():
        raise ValueError("Os centros devem ser coordenadas XYZ finitas.")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("O raio deve ser positivo.")
    if type(max_points) is not int or max_points <= 0:
        raise ValueError("O teto de pontos deve ser um inteiro positivo.")
    xyz = cloud.coordinates_m
    lower, upper = centers.min(axis=0) - radius_m, centers.max(axis=0) + radius_m
    candidates = np.flatnonzero(((xyz >= lower) & (xyz <= upper)).all(axis=1))
    near = np.zeros(len(candidates), dtype=bool)
    for center in centers:
        near |= ((xyz[candidates] - center) ** 2).sum(axis=1) <= radius_m * radius_m
    selected = candidates[near]
    if not len(selected):
        raise ValueError("Nenhum ponto medido dentro do raio pedido.")
    stride = max(1, -(-len(selected) // max_points))
    positions = selected[::stride]
    return (
        PointCloudGeometry(
            xyz[positions],
            cloud.source_indices[positions],
            cloud.map_id,
            cloud.frame_id,
            cloud.source,
            cloud.point_count,
            cloud.data_offset,
            None if cloud.intensities is None else cloud.intensities[positions],
        ),
        stride,
    )


# Contém o formato externo no módulo dono da geometria. A mesma leitura serve
# ao exportador amostrado e à associação sobre todos os pontos medidos.
def read_pcd_geometry(
    source: Path, *, map_id: MapId, frame_id: FrameId, expected_sha256: str | None = None,
) -> PointCloudGeometry:
    """Lê um PCD binário sem subamostrar e valida sua proveniência.

    Levanta:
        ValueError: para schema incompatível, payload truncado ou hash divergente.
    """
    content = source.read_bytes()
    digest = sha256(content).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("O hash do PCD difere da geometria referenciada pelo slice.")
    from io import BytesIO

    stream = BytesIO(content)
    header, offset = _read_header(stream)
    layout, step = _field_layout(header)
    try:
        count = int(header["POINTS"][0])
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError("PCD POINTS must be an integer.") from error
    if count <= 0 or len(content) - offset != count * step:
        raise ValueError("PCD binary payload is truncated or has an invalid size.")
    fields = {}
    for name in ("x", "y", "z", "intensity"):
        if name in layout:
            field_offset, code = layout[name]
            dtype = {"f": "<f4", "d": "<f8", "b": "i1", "B": "u1", "h": "<i2",
                     "H": "<u2", "i": "<i4", "I": "<u4"}[code]
            fields[name] = np.ndarray((count,), dtype=dtype, buffer=content,
                                      offset=offset + field_offset, strides=(step,))
    xyz = np.column_stack([fields[axis] for axis in ("x", "y", "z")]).astype(np.float64)
    finite = np.isfinite(xyz).all(axis=1)
    return PointCloudGeometry(
        xyz[finite], np.flatnonzero(finite), map_id, frame_id,
        SourceArtifactReference(str(source), "application/vnd.pointclouds.pcd", digest),
        count, offset, None if "intensity" not in fields else fields["intensity"][finite],
    )
