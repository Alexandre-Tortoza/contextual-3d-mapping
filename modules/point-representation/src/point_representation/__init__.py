"""Pontos LiDAR convertidos em representações 3D aprendidas.

Fronteira pública: consumidores só devem precisar destes imports. Tudo o
mais dentro de ``point_representation`` é detalhe de implementação.
"""

from __future__ import annotations

from point_representation.application.collate import collate_point_clouds
from point_representation.application.crop import crop_points
from point_representation.application.encoder_batch import collate_encoder_inputs
from point_representation.application.feature_selection import EncoderInput, select_features
from point_representation.application.normalization import denormalize_coordinates, normalize_coordinates
from point_representation.application.voxel_downsample import voxel_downsample
from point_representation.config import (
    CropConfig,
    FeatureSelectionConfig,
    NormalizationConfig,
    VoxelDownsampleConfig,
)
from point_representation.domain.batch import PointBatch
from point_representation.domain.embeddings import BatchedPointEmbeddings, PointEmbedding
from point_representation.domain.encoder_io import EncoderInputBatch, EncoderOutput
from point_representation.domain.errors import FeatureSelectionError, PointCloudValidationError
from point_representation.domain.lineage import PointLineage, compose_lineage
from point_representation.domain.normalization import NormalizationTransform
from point_representation.domain.point_cloud import PointCloud
from point_representation.domain.spatial_bounds import AxisAlignedBounds
from point_representation.domain.training_sample import PointTrainingSample
from point_representation.domain.validation import validate_point_cloud
from point_representation.ports.point_encoder import PointEncoder

__all__ = [
    "AxisAlignedBounds",
    "BatchedPointEmbeddings",
    "CropConfig",
    "EncoderInput",
    "EncoderInputBatch",
    "EncoderOutput",
    "FeatureSelectionConfig",
    "FeatureSelectionError",
    "NormalizationConfig",
    "NormalizationTransform",
    "PointBatch",
    "PointCloud",
    "PointCloudValidationError",
    "PointEmbedding",
    "PointEncoder",
    "PointLineage",
    "PointTrainingSample",
    "VoxelDownsampleConfig",
    "collate_encoder_inputs",
    "collate_point_clouds",
    "compose_lineage",
    "crop_points",
    "denormalize_coordinates",
    "normalize_coordinates",
    "select_features",
    "validate_point_cloud",
    "voxel_downsample",
]
