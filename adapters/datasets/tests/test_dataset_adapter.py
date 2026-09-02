import pytest

from contextual_mapping_adapters import SyntheticDatasetAdapter
from contextual_mapping_datasets import SensorSourceManifest, SequenceManifest
from support import manifest, observation


def test_synthetic_adapter_emits_canonical_observations_in_stable_order() -> None:
    source = SyntheticDatasetAdapter(manifest(), (observation("imu", "imu", 1, 15), observation("lidar", "lidar", 0, 10), observation("rgb", "camera", 0, 10)))
    emitted = tuple(source.observations("sequence"))
    assert [item.reference.observation_id for item in emitted] == ["camera-0", "lidar-0", "imu-1"]
    assert emitted[0].reference.timestamp.nanoseconds == 10
    assert str(emitted[0].reference.frame_id) == "camera"


def test_partial_sequence_with_only_declared_sources_is_valid() -> None:
    source = SyntheticDatasetAdapter(manifest(False), (observation("lidar", "lidar", 0, 10), observation("imu", "imu", 0, 9)))
    assert len(tuple(source.observations("sequence"))) == 2


def test_manifest_rejects_missing_calibration_before_execution() -> None:
    with pytest.raises(ValueError, match="requires calibration_id"):
        SequenceManifest("sequence", (SensorSourceManifest("lidar", "lidar", "file:///lidar", "cloud", "lidar", "clock"),))
