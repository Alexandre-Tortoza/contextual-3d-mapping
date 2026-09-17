"""Treino e inferência públicos do baseline de destilação 2D->3D."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from point_representation.application.encoder_batch import collate_encoder_inputs
from point_representation.application.feature_selection import select_features
from point_representation.config import FeatureSelectionConfig, RidgeDistillationConfig
from point_representation.domain.embeddings import BatchedPointEmbeddings, PointEmbedding
from point_representation.domain.errors import PointCloudValidationError
from point_representation.domain.point_cloud import PointCloud
from point_representation.domain.training_sample import PointTrainingSample
from point_representation.infrastructure.numpy.ridge_distilled_point_encoder import RidgeDistilledPointEncoder
from point_representation.ports.point_encoder import PointEncoder


# Ajusta um baseline de regressão ridge a features de professor já alinhadas a
# pontos. A função é dona da preparação de treino porque teacher_features só
# existe em PointTrainingSample e nunca deve vazar para a API de inferência.
def fit_ridge_distilled_encoder(
    samples: Sequence[PointTrainingSample],
    feature_selection: FeatureSelectionConfig,
    config: RidgeDistillationConfig | None = None,
) -> RidgeDistilledPointEncoder:
    """Treina um encoder 3D linear a partir de supervisão por ponto.

    Argumentos:
        samples: amostras com teacher_features densas, alinhadas a cada ponto.
        feature_selection: canais auxiliares e ordem usados como entrada.
        config: regularização reproduzível da regressão ridge.
    Retorna:
        encoder pronto para inferência pelo port PointEncoder.
    Levanta:
        PointCloudValidationError: se não houver supervisão utilizável ou as
            dimensões de professor divergirem entre amostras.
    """
    if not samples:
        raise PointCloudValidationError("fit_ridge_distilled_encoder requires at least one training sample.")
    if any(sample.teacher_features is None for sample in samples):
        raise PointCloudValidationError("Every training sample must provide per-point teacher_features.")

    inputs = [select_features(sample.point_cloud, feature_selection) for sample in samples]
    batch = collate_encoder_inputs(inputs)
    teachers = [sample.teacher_features for sample in samples]
    assert all(teacher is not None for teacher in teachers)
    teacher_arrays = [
        teacher.reshape(-1, 1) if teacher.ndim == 1 else teacher
        for teacher in teachers
        if teacher is not None
    ]
    dimensions = {teacher.shape[1] for teacher in teacher_arrays}
    if len(dimensions) != 1:
        raise PointCloudValidationError("All teacher_features must have the same embedding dimension.")
    if not dimensions or next(iter(dimensions)) == 0:
        raise PointCloudValidationError("teacher_features must have a non-empty embedding dimension.")

    targets = np.concatenate(teacher_arrays, axis=0)
    if len(targets) == 0:
        raise PointCloudValidationError("fit_ridge_distilled_encoder requires at least one supervised point.")
    raw_inputs = np.concatenate([batch.coordinates, batch.features], axis=1)
    mean = raw_inputs.mean(axis=0)
    scale = raw_inputs.std(axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    normalized = (raw_inputs - mean) / scale
    design = np.concatenate([normalized, np.ones((len(normalized), 1), dtype=np.float64)], axis=1)
    settings = config or RidgeDistillationConfig()
    penalty = np.eye(design.shape[1], dtype=np.float64) * settings.ridge_regularization
    penalty[-1, -1] = 0.0
    system = design.T @ design + penalty
    right_hand = design.T @ targets
    coefficients = (
        np.linalg.solve(system, right_hand)
        if settings.ridge_regularization > 0.0
        else np.linalg.lstsq(design, targets, rcond=None)[0]
    )
    return RidgeDistilledPointEncoder(feature_selection.channels, mean, scale, coefficients)


# Executa a API pública de inferência sobre PointClouds e recompõe a saída do
# port em PointEmbedding por ponto. Pontos mesclados por um encoder genérico
# ficam explicitamente inválidos, preservando a ordem e sem inventar vetores.
def encode_point_clouds(
    encoder: PointEncoder,
    clouds: Sequence[PointCloud],
    feature_selection: FeatureSelectionConfig,
) -> BatchedPointEmbeddings:
    """Codifica nuvens e retorna embeddings na ordem de cada ponto de entrada.

    Argumentos:
        encoder: implementação concreta do port PointEncoder.
        clouds: nuvens de entrada, em ordem de lote.
        feature_selection: canais e ordem solicitados ao encoder.
    Retorna:
        embeddings por ponto, com inválidos explícitos quando a saída funde pontos.
    Levanta:
        PointCloudValidationError: se não houver nuvem ou o lineage for inválido.
    """
    if not clouds:
        raise PointCloudValidationError("encode_point_clouds requires at least one PointCloud.")
    batch = collate_encoder_inputs([select_features(cloud, feature_selection) for cloud in clouds])
    output = encoder.encode(batch)
    assigned: dict[int, np.ndarray] = {}
    merged: set[int] = set()
    for vector, sources in zip(output.features, output.point_lineage.groups, strict=True):
        if len(sources) == 1 and sources[0] not in assigned:
            assigned[sources[0]] = vector
        else:
            merged.update(sources)
    if any(index < 0 or index >= len(batch.coordinates) for group in output.point_lineage.groups for index in group):
        raise PointCloudValidationError("PointEncoder output lineage references an input point outside the batch.")
    embeddings = tuple(
        PointEmbedding(
            coordinates=(float(coordinates[0]), float(coordinates[1]), float(coordinates[2])),
            vector=tuple(float(value) for value in assigned[index]) if index in assigned and index not in merged else (),
            dimension=output.embedding_dimension if index in assigned and index not in merged else 0,
            valid=index in assigned and index not in merged,
        )
        for index, coordinates in enumerate(batch.coordinates)
    )
    return BatchedPointEmbeddings(embeddings=embeddings, sample_offsets=batch.sample_offsets)
