"""Real multimodal reasoning backend adapter.

Issue: #189. Blocked on #174 and a GPU-equipped environment.
"""

from __future__ import annotations

from typing import Any

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import BackendUnavailableError
from visual_perception.domain.image_payload import ImagePayload


class RealMultimodalReasoningAdapter:
    """Satisfies :class:`~visual_perception.ports.multimodal_reasoning.MultimodalReasoner`.

    Not implemented yet. Use
    ``infrastructure.fakes.fake_multimodal_reasoner.FakeMultimodalReasoner``
    for tests and development.
    """

    def analyze_scene(self, image: ImagePayload, config: MultimodalReasoningConfig) -> dict[str, Any]:
        raise BackendUnavailableError(
            f"Multimodal reasoning backend {config.backend!r} is not implemented in this "
            "environment (no GPU, no benchmark-selected checkpoint). See issue #189."
        )

    def analyze_region(
        self,
        image: ImagePayload,
        mask_crop: ImagePayload,
        scene_summary: str | None,
        config: MultimodalReasoningConfig,
    ) -> dict[str, Any]:
        raise BackendUnavailableError(
            f"Multimodal reasoning backend {config.backend!r} is not implemented in this "
            "environment (no GPU, no benchmark-selected checkpoint). See issue #189."
        )
