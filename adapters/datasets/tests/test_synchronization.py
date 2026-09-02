import pytest

from contextual_mapping_adapters import SynchronizationConfig, SyntheticDatasetAdapter, synchronize
from support import manifest, observation


def config(tolerance: int = 5) -> SynchronizationConfig:
    return SynchronizationConfig("lidar", ("lidar", "rgb", "imu", "pose"), tolerance)


def test_exact_near_tolerance_and_missing_observations() -> None:
    group = synchronize((observation("lidar", "lidar", 0, 100), observation("rgb", "camera", 0, 100), observation("imu", "imu", 0, 105)), config())[0]
    assert [item.kind for item in group.observations] == ["lidar", "rgb", "imu"]
    assert group.missing_kinds == ("pose",)
    assert group.observation("imu").reference.timestamp.nanoseconds == 105


def test_out_of_tolerance_is_explicitly_missing() -> None:
    group = synchronize((observation("lidar", "lidar", 0, 100), observation("rgb", "camera", 0, 106)), config())[0]
    assert group.missing_kinds == ("rgb", "imu", "pose")


def test_out_of_order_replay_is_deterministic_and_matches_are_not_reused() -> None:
    items = (observation("lidar", "lidar", 0, 100), observation("rgb", "camera", 0, 102), observation("lidar", "lidar", 1, 104))
    policy = SynchronizationConfig("lidar", ("lidar", "rgb"), 5)
    first = synchronize(items, policy)
    assert first == synchronize(reversed(items), policy)
    assert first[0].observation("rgb") is not None
    assert first[1].missing_kinds == ("rgb",)


def test_invalid_tolerance_fails_before_execution() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        config(-1)


def test_adapter_output_synchronizes_without_dataset_specific_types() -> None:
    adapter = SyntheticDatasetAdapter(manifest(), (observation("imu", "imu", 0, 99), observation("rgb", "camera", 0, 101), observation("lidar", "lidar", 0, 100)))
    groups = synchronize(adapter.observations("sequence"), config())
    assert len(groups) == 1
    assert groups[0].missing_kinds == ("pose",)
