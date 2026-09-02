from contextual_mapping_adapters import CanonicalObservation
from contextual_mapping_contracts import FrameId, ObservationReference, SourceArtifactReference, Timestamp
from contextual_mapping_datasets import CalibrationManifest, DatasetManifest, SensorSourceManifest, SequenceManifest


def manifest(include_rgb: bool = True) -> DatasetManifest:
    calibration = CalibrationManifest("cal", "file:///cal.json", "sensor", "base")
    sensors = [
        SensorSourceManifest("lidar", "lidar", "file:///lidar", "cloud", "lidar", "clock", "cal"),
        SensorSourceManifest("imu", "imu", "file:///imu", "text/csv", "imu", "clock", "cal"),
    ]
    if include_rgb:
        sensors.append(SensorSourceManifest("camera", "rgb", "file:///rgb", "image/png", "camera", "clock", "cal"))
    return DatasetManifest("dataset", (SequenceManifest("sequence", tuple(sensors), (calibration,)),))


def observation(kind: str, sensor: str, index: int, timestamp_ns: int) -> CanonicalObservation:
    return CanonicalObservation(
        kind=kind,
        reference=ObservationReference(f"{sensor}-{index}", "dataset", "sequence", sensor, index, Timestamp(timestamp_ns, "clock"), FrameId(sensor), "cal"),
        artifact=SourceArtifactReference(f"file:///{sensor}/{index}", "application/octet-stream"),
    )
