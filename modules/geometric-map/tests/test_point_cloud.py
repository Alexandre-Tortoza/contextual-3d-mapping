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
