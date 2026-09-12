"""Regressões de primeira superfície, aberturas e amostragem do viewer."""

from dataclasses import replace

import numpy as np
import pytest
from geometric_map import GeometryReference, PointCloudGeometry
from sensor_association import (
    AssociationStatus,
    BoundaryPolicy,
    CameraLidarCalibration,
    CameraModel,
    MapAnchoredPoint,
    MeasuredSurfaceModel,
    RgbFrame,
    SurfaceVisibilityConfig,
    VisualRegionEvidence,
    associate_measured_map_points,
)
from sensor_association.camera_geometry import pixel_rays, project_coordinates

from contextual_mapping_contracts import (
    FrameId,
    MapId,
    ObservationReference,
    RigidTransform,
    SourceArtifactReference,
    Timestamp,
)


# Cria medições determinísticas no frame de câmera, sem depender de arquivos reais.
def _geometry(xyz):
    """Constrói uma nuvem com identidade e índices preservados."""
    xyz = np.asarray(xyz)
    return PointCloudGeometry(xyz, np.arange(len(xyz)), MapId('map'), FrameId('map'), SourceArtifactReference('fixture://pcd', 'application/pcd'), len(xyz))


# Produz planos densos para controlar inclinação, distância e aberturas.
def _plane(z, extent=2, step=.05, slope=0):
    """Amostra uma superfície plana com espaçamento declarado em metros."""
    x, y = np.meshgrid(np.arange(-extent, extent+step/2, step), np.arange(-extent, extent+step/2, step))
    return np.column_stack((x.ravel(), y.ravel(), (z + slope*x).ravel()))


# Fornece uma câmera pinhole simples e relógios sincronizados na fronteira pública.
def _camera():
    """Retorna RGB, calibração, pose e âncora LiDAR da fixture."""
    ref = ObservationReference('rgb', 'dataset', 'sequence', 'camera', 0, Timestamp(100, 'clock'), FrameId('camera'))
    transform = RigidTransform(FrameId('map'), FrameId('camera'), (0,0,0), (0,0,0,1))
    cal = CameraLidarCalibration('cal', SourceArtifactReference('fixture://cal', 'application/json'), CameraModel.PINHOLE, 50,50,40,40, replace(transform, source_frame=FrameId('lidar')))
    rgb = RgbFrame(ref,81,81,((120,140,160),)*81**2,frozenset((x,y) for y in range(81) for x in range(81)))
    return rgb, cal, transform, replace(ref, observation_id='lidar', frame_id=FrameId('lidar'), sensor_id='lidar')


# Encaminha pontos escolhidos à associação real; a geometria de referência é fixa.
def _associate(xyz, targets, regions=()):
    """Associa candidatos identificados usando patches medidos completos."""
    rgb, cal, pose, lidar = _camera()
    points = tuple(MapAnchoredPoint(GeometryReference(MapId('map'), str(i)), tuple(point)) for i, point in enumerate(targets))
    return associate_measured_map_points(points, rgb, cal, pose, MeasuredSurfaceModel(_geometry(xyz)), regions, lidar_observation=lidar, boundary_policy=BoundaryPolicy(enabled=False))


# Protege a inversão usada pelo raster contra divergências em câmeras de campo amplo.
@pytest.mark.parametrize('model', list(CameraModel))
def test_calibrated_ray_roundtrip(model):
    """Projeção e inversão retornam o mesmo raio nos três modelos públicos."""
    _, cal, _, _ = _camera()
    cal = replace(cal, model=model, mirror_xi=1.8 if model is CameraModel.MEI else None,
                  distortion_k1=.08, distortion_k2=-.015, distortion_p1=.001, distortion_p2=-.002)
    rays = np.array([[0,0,1],[.4,.2,1],[-.9,.3,1],[2,-1,.25]], dtype=float)
    rays /= np.linalg.norm(rays,axis=1)[:,None]
    np.testing.assert_allclose(pixel_rays(project_coordinates(rays,cal),cal), rays, atol=1e-7)


# Reproduz a parede distante alcançada pelo mesmo raio de uma superfície próxima.
def test_first_surface_rejects_far_wall_in_l_corridor():
    """A parede ao fundo não herda RGB de uma superfície frontal."""
    xyz = np.concatenate((_plane(2), _plane(12, extent=8,step=.2)))
    result = _associate(xyz, [(0,0,2),(0,0,12)])
    assert [a.status for a in result.associations] == [AssociationStatus.ASSOCIATED, AssociationStatus.OCCLUDED]
    assert result.associations[1].surface_evidence.first_surface_depth_m == pytest.approx(2)


# A profundidade correta de um plano inclinado varia entre raios vizinhos.
def test_tilted_plane_does_not_occlude_itself():
    """Interseções locais preservam pontos próximos em um plano inclinado."""
    targets = [(float(x),0.,2.+float(x)) for x in np.linspace(-.5,1,20)]
    result = _associate(_plane(2,extent=1,slope=1), targets)
    assert all(a.status is AssociationStatus.ASSOCIATED for a in result.associations)


# O cilindro fornece duas raízes positivas e um caso próximo da tangência.
def test_cylinder_uses_near_intersection_and_abstains_at_unstable_tangent():
    """O lado de trás nunca vence a primeira interseção do cilindro."""
    angle, y = np.meshgrid(np.linspace(0,2*np.pi,721),np.arange(-.3,.31,.03))
    xyz = np.column_stack((np.sin(angle).ravel(), y.ravel(), (3+np.cos(angle)).ravel()))
    model = MeasuredSurfaceModel(_geometry(xyz), SurfaceVisibilityConfig(max_patch_radius_m=.06))
    view = model.camera_view(_camera()[2])
    ray = np.array([[0.,0.,1.],[.30,0.,1.]])
    ray /= np.linalg.norm(ray,axis=1)[:,None]
    hit = view.intersect(ray)
    expected = 3*ray[:,2] - np.sqrt((3*ray[:,2])**2 - 8)
    np.testing.assert_allclose(hit.ranges_m, expected, atol=.025)
    tangent = np.array([[1.,0.,np.sqrt(8)]])/3
    tangent_hit = view.intersect(tangent)
    assert np.isinf(tangent_hit.ranges_m[0]) or abs(tangent_hit.ranges_m[0]-np.sqrt(8)) < .15


# A superfície da paisagem pode ser visível sem representar a janela reconhecida.
def test_window_frame_is_supported_but_landscape_is_only_rgb():
    """Mantém o aro medido e impede que o fundo atravesse o vínculo semântico."""
    wall = _plane(2)
    wall = wall[(np.abs(wall[:,0])>.4) | (np.abs(wall[:,1])>.4)]
    geometry = np.concatenate((wall,_plane(6,extent=5,step=.1)))
    region = VisualRegionEvidence('window', frozenset((x,y) for y in range(20,61) for x in range(20,61)), 'window', grounding_status='refined', grounding_reference='fixture://mask')
    result = _associate(geometry, [(.6,0,2),(0,0,6)], (region,))
    frame, landscape = result.associations
    assert frame.label == 'window'
    assert landscape.status is AssociationStatus.ASSOCIATED
    assert landscape.label is None and landscape.tentative_label == 'window'
    assert landscape.surface_evidence.surface_support == 'uncertain'


# Uma máscara sozinha não demonstra que o único retorno seja a superfície do objeto.
def test_window_without_measured_frame_remains_2d_evidence():
    """A ausência de suporte frontal não promove o fundo nem cria um plano."""
    region = VisualRegionEvidence('window', frozenset((x,y) for y in range(20,61) for x in range(20,61)), 'window', grounding_status='refined', grounding_reference='fixture://mask')
    result = _associate(_plane(6,extent=5,step=.1), [(0,0,6)], (region,))
    assert result.associations[0].status is AssociationStatus.ASSOCIATED
    assert result.associations[0].label is None
    assert result.regions['window']['status'] == 'uncertain'


# Falta de geometria não é autorização para associar o ponto mais distante.
def test_sparse_geometry_abstains():
    """Menos de seis medidas não sustentam um patch planar."""
    result = _associate([(0,0,2),(.1,0,2),(0,.1,2)], [(0,0,2)])
    assert result.associations[0].status is AssociationStatus.VISIBILITY_UNCONFIRMED
    assert result.associations[0].label is None


# Pontos de renderização não participam da construção dos oclusores ou do vínculo.
def test_common_point_decision_is_invariant_to_viewer_sampling():
    """Subamostrar e reordenar candidatos mantém inclusive os diagnostics."""
    rgb, cal, pose, lidar = _camera()
    model = MeasuredSurfaceModel(_geometry(np.concatenate((_plane(2),_plane(6)))))
    points = tuple(MapAnchoredPoint(GeometryReference(MapId('map'), str(i)), p) for i,p in enumerate(((0.,0.,2.),(.01,0.,2.),(0.,0.,6.))))
    full = associate_measured_map_points(points,rgb,cal,pose,model,lidar_observation=lidar).associations
    subset = associate_measured_map_points((points[2],points[1]),rgb,cal,pose,model,lidar_observation=lidar).associations
    assert subset == (full[2],full[1])
    assert full[0].status is full[1].status is AssociationStatus.ASSOCIATED


# Um mapa ou frame diferente não pode ser usado como oclusor por coincidência numérica.
def test_visibility_geometry_frame_and_identity_are_validated():
    """Falha cedo quando a proveniência espacial é incompatível."""
    rgb, cal, pose, lidar = _camera()
    model = MeasuredSurfaceModel(_geometry(_plane(2)))
    point = MapAnchoredPoint(GeometryReference(MapId('other'), '0'),(0.,0.,2.))
    with pytest.raises(ValueError,match='map_id'):
        associate_measured_map_points((point,),rgb,cal,pose,model,lidar_observation=lidar)
    with pytest.raises(ValueError,match='frame'):
        model.camera_view(replace(pose, source_frame=FrameId('other')))


# Densidade angular do fundo não pode expulsar os patches próximos da consulta.
def test_dense_background_does_not_hide_sparse_foreground_occluder():
    """Encontra a superfície frontal mesmo com muito mais medições ao fundo."""
    geometry=np.concatenate((_plane(2,extent=.3,step=.1),_plane(8,extent=1,step=.015)))
    result=_associate(geometry,[(.12,.04,8)])
    assert result.associations[0].status is AssociationStatus.OCCLUDED
    assert result.associations[0].surface_evidence.first_surface_depth_m < 2.01


# Uma região pode conter quase somente fundo sem autorização para rotulá-lo.
def test_background_dominance_does_not_transfer_the_window_label():
    """O fundo majoritário continua sem label, com uma moldura medida estreita."""
    wall=_plane(2)
    wall=wall[(np.abs(wall[:,0])>.65)|(np.abs(wall[:,1])>.65)]
    geometry=np.concatenate((wall,_plane(6,extent=5,step=.1)))
    region=VisualRegionEvidence('opening',frozenset((x,y)for y in range(20,61)for x in range(20,61)),
        'opening',grounding_status='refined',grounding_reference='fixture://mask')
    result=_associate(geometry,[(0,0,6),(.5,.2,6),(.75,0,2)],(region,))
    assert all(item.label is None for item in result.associations[:2])
    assert result.associations[2].label == 'opening'
    assert result.regions['opening']['supported_pixels'] < result.regions['opening']['measured_pixels']/2


# Um patch finito não deve bloquear raios que passam ao lado de sua borda.
def test_ray_outside_bounded_foreground_reaches_far_surface():
    """Respeita o fim da superfície próxima ao dobrar um corredor."""
    geometry=np.concatenate((_plane(2,extent=.3,step=.05),_plane(6,extent=3,step=.1)))
    result=_associate(geometry,[(2.,0.,6.)])
    assert result.associations[0].status is AssociationStatus.ASSOCIATED
    assert result.associations[0].surface_evidence.first_surface_depth_m == pytest.approx(np.sqrt(40))


# A extrínseca referenciada deve descrever os sensores da associação auditada.
@pytest.mark.parametrize('side', ['source_frame', 'target_frame'])
def test_rejects_calibration_for_other_sensor_frames(side):
    """Recusa calibração de outro sensor mesmo com intrínsecos iguais."""
    rgb,cal,pose,lidar=_camera()
    cal=replace(cal,lidar_to_camera=replace(cal.lidar_to_camera,**{side:FrameId('other')}))
    with pytest.raises(ValueError,match='calibration .* frame'):
        associate_measured_map_points((),rgb,cal,pose,MeasuredSurfaceModel(_geometry(_plane(2))),lidar_observation=lidar)
