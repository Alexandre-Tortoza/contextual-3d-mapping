"""Schema público de manifest de dataset e de conjunto de referência anotado."""

from .annotation_manifest import (
    REFERENCE_SCHEMA_VERSION,
    AnnotationCertainty,
    AnnotationProvenance,
    MaskAnnotation,
    ReferenceManifest,
    RegionAnnotation,
    RelationAnnotation,
    ReviewState,
    SampleAnnotation,
    SampleArtifact,
    Split,
    validate_reference,
)
from .annotation_manifest_io import (
    load_reference_manifest,
    reference_manifest_from_mapping,
    reference_manifest_to_mapping,
    save_reference_manifest,
)
from .manifest import CalibrationManifest, DatasetManifest, SensorSourceManifest, SequenceManifest
from .manifest_io import dataset_manifest_from_mapping, load_dataset_manifest
from .paths import RAW_DATASETS_DIRECTORY, raw_dataset_root, validate_dataset_name

__all__ = [
    "RAW_DATASETS_DIRECTORY",
    "REFERENCE_SCHEMA_VERSION",
    "AnnotationCertainty",
    "AnnotationProvenance",
    "CalibrationManifest",
    "DatasetManifest",
    "MaskAnnotation",
    "ReferenceManifest",
    "RegionAnnotation",
    "RelationAnnotation",
    "ReviewState",
    "SampleAnnotation",
    "SampleArtifact",
    "SensorSourceManifest",
    "SequenceManifest",
    "Split",
    "dataset_manifest_from_mapping",
    "load_dataset_manifest",
    "load_reference_manifest",
    "raw_dataset_root",
    "reference_manifest_from_mapping",
    "reference_manifest_to_mapping",
    "save_reference_manifest",
    "validate_dataset_name",
    "validate_reference",
]
