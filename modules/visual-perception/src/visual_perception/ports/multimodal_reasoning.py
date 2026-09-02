"""Multimodal reasoning boundary for scene- and region-level interpretation.

Issue: #189 (real adapter shape). Used by the application stages in #164
(scene) and #165 (region). The port returns a raw structured response;
schema parsing and semantic validation stay in the application layer, not in
backend-specific transport code (see #189's scope).
"""

from __future__ import annotations

from typing import Any, Protocol

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.image_payload import ImagePayload


class MultimodalReasoner(Protocol):
    """Runs scene-level and region-level structured prompts on one backend."""

    def analyze_scene(
        self, image: ImagePayload, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Return a raw structured response describing the whole scene.

        Malformed/unparseable responses are the application stage's concern
        (#164); this method should raise
        :class:`~visual_perception.domain.errors.BackendExecutionError` only
        for backend/transport failures.
        """
        ...

    def analyze_region(
        self,
        image: ImagePayload,
        mask_crop: ImagePayload,
        scene_summary: str | None,
        config: MultimodalReasoningConfig,
    ) -> dict[str, Any]:
        """Return a raw structured response describing one region (#165)."""
        ...
