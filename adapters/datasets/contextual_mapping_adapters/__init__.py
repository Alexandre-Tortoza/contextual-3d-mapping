"""Fronteira de entrada de dataset normalizada e sincronização determinística."""

from .dataset import CanonicalObservation, MultimodalDatasetAdapter, SyntheticDatasetAdapter
from .filesystem import DatasetFilesystem
from .rosbag_images import (
    ROSBAG_CLOCK_ID,
    ExtractedFrameProvenance,
    RosbagImageTopic,
    RosbagRecording,
    decode_image_message,
    extract_rosbag_frames,
    extract_strided_rosbag_frames,
    extract_uniform_rosbag_frames,
    frame_provenance_path,
    inspect_rosbag,
    read_frame_provenance,
    select_rgb_topic,
    write_frame_provenance,
)
from .synchronization import SynchronizationConfig, SynchronizedObservationGroup, synchronize

__all__ = [
    "ROSBAG_CLOCK_ID",
    "CanonicalObservation",
    "DatasetFilesystem",
    "ExtractedFrameProvenance",
    "MultimodalDatasetAdapter",
    "RosbagImageTopic",
    "RosbagRecording",
    "SynchronizationConfig",
    "SynchronizedObservationGroup",
    "SyntheticDatasetAdapter",
    "decode_image_message",
    "extract_rosbag_frames",
    "extract_strided_rosbag_frames",
    "extract_uniform_rosbag_frames",
    "frame_provenance_path",
    "inspect_rosbag",
    "read_frame_provenance",
    "select_rgb_topic",
    "synchronize",
    "write_frame_provenance",
]
