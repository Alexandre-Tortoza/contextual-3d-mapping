"""Integrate canonical perception execution with mapping-runtime.

Issue: #179.

``mapping-runtime`` (#106) is not implemented yet. This module shows and
tests the intended integration shape: the runtime calls only
``visual_perception``'s public boundary (``run_canonical_pipeline`` and its
public types), never a private implementation path, and receives module
failures as an explicit diagnostic instead of a raw exception leaking a
backend-specific type.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception import ImageObservation, ImagePayload, ModuleConfig, PerceptionPorts, VisualObservation
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.domain.errors import VisualPerceptionError


@dataclass(frozen=True)
class RuntimeDiagnostic:
    """A module failure surfaced to the runtime, never a raw backend exception."""

    observation_id: str
    stage: str
    message: str


def run_visual_perception_for_runtime(
    image: ImageObservation, payload: ImagePayload, config: ModuleConfig, ports: PerceptionPorts
) -> VisualObservation | RuntimeDiagnostic:
    """Run the canonical pipeline the way ``mapping-runtime`` is expected to call it."""
    try:
        result = run_canonical_pipeline(image, payload, config, ports)
    except VisualPerceptionError as error:
        return RuntimeDiagnostic(
            observation_id=image.observation_id, stage="visual_perception", message=str(error)
        )
    return result.observation
