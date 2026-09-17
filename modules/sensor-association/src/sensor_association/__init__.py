"""Fronteira pública do módulo sensor-association."""

from .boundary import BoundaryPolicy, boundary_diagnostics, distance_to_mask_boundary
from .dense_features import DenseFeatureMap, sample_dense_feature
from .measured_association import SurfaceAssociationResult, associate_measured_map_points
from .models import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    MapAnchoredPoint,
    PointVisualAssociation,
    RgbFrame,
    SemanticAssociationStatus,
    SurfaceAssociationEvidence,
    VisualRegionEvidence,
)
from .projector import associate_map_points, associate_points
from .surface_visibility import MeasuredSurfaceModel, SurfaceVisibilityConfig

__all__ = [
    "AssociationStatus",
    "CameraLidarCalibration",
    "CameraModel",
    "MapAnchoredPoint",
    "PointVisualAssociation",
    "RgbFrame",
    "VisualRegionEvidence",
    "associate_map_points",
    "associate_points",
    "BoundaryPolicy",
    "SemanticAssociationStatus",
    "boundary_diagnostics",
    "distance_to_mask_boundary",
    "SurfaceAssociationEvidence",
    "SurfaceAssociationResult",
    "MeasuredSurfaceModel",
    "SurfaceVisibilityConfig",
    "associate_measured_map_points",
    "DenseFeatureMap",
    "sample_dense_feature",
]
