"""Testes de contract de PointEmbedding/BatchedPointEmbeddings (#1)."""

from __future__ import annotations

import pytest

from point_representation import BatchedPointEmbeddings, PointEmbedding


def _embedding(valid: bool = True) -> PointEmbedding:
    return PointEmbedding((0.0, 1.0, 2.0), (0.1, 0.2, 0.3), 3, valid=valid)


def test_valid_embedding_round_trips_fields() -> None:
    embedding = _embedding()

    assert embedding.coordinates == (0.0, 1.0, 2.0)
    assert embedding.vector == (0.1, 0.2, 0.3)


def test_invalid_embedding_does_not_require_a_vector() -> None:
    embedding = PointEmbedding((0.0, 0.0, 0.0), (), 0, valid=False)

    assert embedding.valid is False


def test_dimension_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        PointEmbedding((0.0, 0.0, 0.0), (0.1, 0.2), 3, valid=True)


def test_empty_vector_while_valid_is_rejected() -> None:
    with pytest.raises(ValueError):
        PointEmbedding((0.0, 0.0, 0.0), (), 0, valid=True)


def test_non_finite_vector_is_rejected() -> None:
    with pytest.raises(ValueError):
        PointEmbedding((0.0, 0.0, 0.0), (float("nan"), 0.0), 2, valid=True)


def test_confidence_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        PointEmbedding((0.0, 0.0, 0.0), (0.1,), 1, confidence=1.5)


def test_batched_embeddings_splits_per_sample() -> None:
    embeddings = tuple(_embedding() for _ in range(5))
    batch = BatchedPointEmbeddings(embeddings, sample_offsets=(0, 2, 5))

    grouped = batch.per_sample()

    assert len(grouped) == 2
    assert grouped[0] == embeddings[:2]
    assert grouped[1] == embeddings[2:]


def test_batch_offsets_must_start_at_zero() -> None:
    with pytest.raises(ValueError):
        BatchedPointEmbeddings((_embedding(),), sample_offsets=(1,))


def test_batch_offsets_must_end_at_embedding_count() -> None:
    with pytest.raises(ValueError):
        BatchedPointEmbeddings((_embedding(), _embedding()), sample_offsets=(0, 1))


def test_batch_offsets_must_be_non_decreasing() -> None:
    with pytest.raises(ValueError):
        BatchedPointEmbeddings((_embedding(), _embedding()), sample_offsets=(0, 2, 1))
