"""Fronteira de entrada de dataset normalizada e sincronização determinística."""

from .dataset import CanonicalObservation, MultimodalDatasetAdapter, SyntheticDatasetAdapter
from .filesystem import DatasetFilesystem
from .rosbag_images import (
    RosbagImageTopic,
    RosbagRecording,
    decode_image_message,
    extract_rosbag_frames,
    extract_uniform_rosbag_frames,
    inspect_rosbag,
    select_rgb_topic,
)
from .synchronization import SynchronizationConfig, SynchronizedObservationGroup, synchronize

__all__ = [
    "CanonicalObservation",
    "DatasetFilesystem",
    "MultimodalDatasetAdapter",
    "RosbagImageTopic",
    "RosbagRecording",
    "SynchronizationConfig",
    "SynchronizedObservationGroup",
    "SyntheticDatasetAdapter",
    "decode_image_message",
    "extract_rosbag_frames",
    "extract_uniform_rosbag_frames",
    "inspect_rosbag",
    "select_rgb_topic",
    "synchronize",
]
