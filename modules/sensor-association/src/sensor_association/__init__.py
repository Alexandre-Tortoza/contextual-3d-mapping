"""Fronteira pública do módulo sensor-association."""

from .models import (
    AssociationStatus,
    CameraLidarCalibration,
    CameraModel,
    PointVisualAssociation,
    RgbFrame,
    VisualRegionEvidence,
)
from .projector import associate_points

__all__ = [
    "AssociationStatus",
    "CameraLidarCalibration",
    "CameraModel",
    "PointVisualAssociation",
    "RgbFrame",
    "VisualRegionEvidence",
    "associate_points",
]
