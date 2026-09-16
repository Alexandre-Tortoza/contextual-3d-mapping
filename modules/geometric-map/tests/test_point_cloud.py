"""Contract de leitura integral de PCD e proveniência da geometria."""

import struct
from hashlib import sha256

import numpy as np
import pytest
from geometric_map import read_pcd_geometry

from contextual_mapping_contracts import FrameId, MapId


# Inclui um campo vetorial antes de XYZ para proteger os offsets e os índices.
def _pcd(path):
    """Grava três registros, incluindo uma medição XYZ inválida."""
    header = b'FIELDS descriptor x y z intensity\nSIZE 4 4 4 4 2\nTYPE F F F F U\nCOUNT 2 1 1 1 1\nPOINTS 3\nDATA binary\n'
    records = [(9,8,1,2,3,42),(9,8,float('nan'),2,3,51),(9,8,4,5,6,63)]
    path.write_bytes(header+b''.join(struct.pack('<fffffH',*record) for record in records))
    return sha256(path.read_bytes()).hexdigest()


# O viewer e a visibilidade precisam concordar com os mesmos IDs de origem.
def test_full_geometry_preserves_source_indices_and_hash(tmp_path):
    """Remove NaNs sem renumerar as medições e protege arrays contra escrita."""
    path=tmp_path/'cloud.pcd'
    digest=_pcd(path)
    cloud=read_pcd_geometry(path,map_id=MapId('map'),frame_id=FrameId('map'),expected_sha256=digest)
    np.testing.assert_array_equal(cloud.source_indices,[0,2])
    np.testing.assert_array_equal(cloud.coordinates_m,[[1,2,3],[4,5,6]])
    np.testing.assert_array_equal(cloud.intensities,[42,63])
    assert cloud.source.digest == digest and cloud.point_count == 3
    with pytest.raises(ValueError):
        cloud.coordinates_m[0,0]=7


# Hash e payload são verificados antes de uma geometria servir como oclusor.
def test_rejects_mismatched_or_truncated_source(tmp_path):
    """Falha explicitamente para proveniência diferente ou bytes ausentes."""
    path=tmp_path/'cloud.pcd'
    _pcd(path)
    with pytest.raises(ValueError,match='hash'):
        read_pcd_geometry(path,map_id=MapId('map'),frame_id=FrameId('map'),expected_sha256='different')
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError,match='truncated'):
        read_pcd_geometry(path,map_id=MapId('map'),frame_id=FrameId('map'))


# Cria uma nuvem sintética em linha para exercitar a seleção por vizinhança.
def _line_cloud(count=100):
    """Retorna pontos espaçados de 1 m no eixo X com índices de origem esparsos."""
    from contextual_mapping_contracts import SourceArtifactReference
    from geometric_map import PointCloudGeometry

    xyz = np.column_stack([np.arange(count, dtype=float), np.zeros(count), np.zeros(count)])
    source = SourceArtifactReference(uri="mem://line.pcd", media_type="application/vnd.pointclouds.pcd", digest="0" * 64)
    return PointCloudGeometry(xyz, np.arange(count) * 3, MapId("map"), FrameId("map"), source, count * 3, intensities=np.arange(count, dtype=float))


# O trecho carrega só a vizinhança das poses, com os índices de origem intactos, e
# o teto é respeitado por amostragem determinística dentro da seleção.
def test_neighbourhood_keeps_points_near_any_center_and_source_indices():
    """Seleciona pontos a até o raio de algum centro e aplica o teto por passo."""
    from geometric_map import select_neighbourhood

    cloud = _line_cloud()
    selected, stride = select_neighbourhood(
        cloud, np.array([[10.0, 0, 0], [80.0, 0, 0]]), centers_frame=FrameId("map"), radius_m=2.0, max_points=100
    )
    assert stride == 1
    np.testing.assert_array_equal(selected.coordinates_m[:, 0], [8, 9, 10, 11, 12, 78, 79, 80, 81, 82])
    np.testing.assert_array_equal(selected.source_indices, selected.coordinates_m[:, 0].astype(int) * 3)
    np.testing.assert_array_equal(selected.intensities, selected.coordinates_m[:, 0])

    capped, stride = select_neighbourhood(
        cloud, np.array([[50.0, 0, 0]]), centers_frame=FrameId("map"), radius_m=20.0, max_points=10
    )
    assert stride == 5 and len(capped.coordinates_m) <= 10
    assert capped.point_count == cloud.point_count


# Parâmetros inválidos e um raio vazio falham em vez de produzir um trecho sem geometria.
def test_neighbourhood_rejects_invalid_requests():
    """Recusa centros, raio e teto inválidos e seleção vazia."""
    from geometric_map import select_neighbourhood

    cloud = _line_cloud()
    for centers, radius, cap in ((np.zeros((0, 3)), 1.0, 10), (np.zeros((1, 3)), 0.0, 10), (np.zeros((1, 3)), 1.0, 0)):
        with pytest.raises(ValueError):
            select_neighbourhood(cloud, centers, centers_frame=FrameId("map"), radius_m=radius, max_points=cap)
    with pytest.raises(ValueError, match="Nenhum ponto"):
        select_neighbourhood(
            cloud, np.array([[0.0, 50.0, 0.0]]), centers_frame=FrameId("map"), radius_m=1.0, max_points=10
        )


# Centros expressos em outro frame não devem ser aceitos silenciosamente: um recorte
# de vizinhança calculado sobre um frame diferente do da nuvem selecionaria pontos
# sem qualquer garantia espacial real.
def test_neighbourhood_rejects_mismatched_frame():
    """Recusa centros cujo frame declarado diverge do frame da nuvem."""
    from geometric_map import select_neighbourhood

    cloud = _line_cloud()
    with pytest.raises(ValueError, match="frame"):
        select_neighbourhood(
            cloud, np.array([[10.0, 0, 0]]), centers_frame=FrameId("odom"), radius_m=2.0, max_points=100
        )
