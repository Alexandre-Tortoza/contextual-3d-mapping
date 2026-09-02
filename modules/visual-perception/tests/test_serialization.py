"""Serialization round-trip tests (#172)."""

from __future__ import annotations

import pytest

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.infrastructure.serialization import (
    UnsupportedSchemaVersionError,
    deserialize_observation,
    serialize_observation,
)


def test_round_trip_preserves_regions_claims_and_relations() -> None:
    payload = payload_with_blobs(
        blobs=((2, 2, 8, 8, (200, 30, 30)), (20, 20, 28, 28, (30, 200, 30)))
    )
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())

    round_tripped = deserialize_observation(serialize_observation(result.observation))

    assert round_tripped == result.observation


def test_unsupported_schema_version_fails_predictably() -> None:
    payload = serialize_observation(
        run_canonical_pipeline(
            image_observation(), payload_with_blobs(), default_config(), default_ports()
        ).observation
    )
    payload["schema_version"] = 999
    with pytest.raises(UnsupportedSchemaVersionError):
        deserialize_observation(payload)
