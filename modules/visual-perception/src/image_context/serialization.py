"""JSON serialization for resumable pipeline artifacts."""

from __future__ import annotations

from typing import Any

from image_context.models import ImageSample


def sample_to_dict(sample: ImageSample) -> dict[str, Any]:
    """Serialize an extracted sample."""
    return {
        "frame_id": sample.frame_id,
        "source_index": sample.source_index,
        "timestamp_ns": sample.timestamp_ns,
        "width": sample.width,
        "height": sample.height,
        "image_path": str(sample.image_path),
    }
