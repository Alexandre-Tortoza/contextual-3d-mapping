"""Fronteira pública do módulo sensor-association."""

from .models import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    MapAnchoredPoint,
    PointVisualAssociation,
    RgbFrame,
    VisualRegionEvidence,
)
from .projector import associate_map_points, associate_points

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
]
