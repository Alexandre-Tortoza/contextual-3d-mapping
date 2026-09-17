"""Encoder 3D linear treinado por destilação de features por ponto."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from point_representation.domain.encoder_io import EncoderInputBatch, EncoderOutput
from point_representation.domain.errors import FeatureSelectionError
from point_representation.domain.lineage import PointLineage


# Materializa o primeiro encoder 3D concreto do módulo. Ele é deliberadamente
# linear e framework-agnóstico: serve como baseline reproduzível para verificar
# se a supervisão visual 2D->3D carrega sinal antes de aumentar a capacidade do
# backbone; é criado exclusivamente por fit_ridge_distilled_encoder().
@dataclass(frozen=True)
class RidgeDistilledPointEncoder:
    """Encoder por ponto ajustado para reproduzir features de professor.

    Argumentos:
        selected_channels: canais auxiliares esperados, em ordem canônica.
        input_mean: média das coordenadas e canais observados no treino.
        input_scale: escala positiva usada para normalizar cada entrada.
        coefficients: pesos incluindo a última linha de bias.
    """

    selected_channels: tuple[str, ...]
    input_mean: np.ndarray
    input_scale: np.ndarray
    coefficients: np.ndarray

    # Valida o estado treinado na fronteira do adapter para que um checkpoint
    # incompleto nunca produza embeddings de dimensão ou canais silenciosamente
    # incorretos durante inferência.
    def __post_init__(self) -> None:
        """Valida dimensões, finitude e escalas do estado treinado."""
        mean = np.asarray(self.input_mean, dtype=np.float64)
        scale = np.asarray(self.input_scale, dtype=np.float64)
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if mean.ndim != 1 or scale.shape != mean.shape:
            raise ValueError("RidgeDistilledPointEncoder input_mean and input_scale must be aligned 1D arrays.")
        if coefficients.ndim != 2 or coefficients.shape[0] != len(mean) + 1:
            raise ValueError("RidgeDistilledPointEncoder coefficients must have input_dimension + 1 rows.")
        if coefficients.shape[1] == 0:
            raise ValueError("RidgeDistilledPointEncoder requires a non-empty teacher embedding dimension.")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coefficients).all():
            raise ValueError("RidgeDistilledPointEncoder state must contain only finite values.")
        if (scale <= 0.0).any():
            raise ValueError("RidgeDistilledPointEncoder input_scale must be strictly positive.")
        for field_name, value in (("input_mean", mean), ("input_scale", scale), ("coefficients", coefficients)):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, field_name, copied)

    @property
    def embedding_dimension(self) -> int:
        """Retorna a dimensão das features de professor destiladas."""
        return int(self.coefficients.shape[1])

    @property
    def required_channels(self) -> frozenset[str]:
        """Retorna os canais de entrada que o encoder foi treinado para consumir."""
        return frozenset(self.selected_channels)

    # Aplica o modelo ajustado a um batch, mantendo um embedding por ponto e a
    # identidade de índices porque este baseline não discretiza nem mescla a
    # geometria. É a implementação concreta do port PointEncoder.
    def encode(self, batch: EncoderInputBatch) -> EncoderOutput:
        """Codifica um lote usando a regressão ridge destilada.

        Argumentos:
            batch: coordenadas e canais na mesma ordem usada no treino.
        Retorna:
            features destiladas, uma por ponto de entrada.
        Levanta:
            FeatureSelectionError: se os canais ou sua ordem forem incompatíveis.
            ValueError: se a largura de entrada não corresponder ao encoder.
        """
        if batch.selected_channels != self.selected_channels:
            raise FeatureSelectionError(
                "RidgeDistilledPointEncoder requires selected channels "
                f"{self.selected_channels}, got {batch.selected_channels}."
            )
        inputs = np.concatenate([batch.coordinates, batch.features], axis=1)
        if inputs.shape[1] != len(self.input_mean):
            raise ValueError(
                "RidgeDistilledPointEncoder input dimension does not match its trained state: "
                f"got {inputs.shape[1]}, expected {len(self.input_mean)}."
            )
        normalized = (inputs - self.input_mean) / self.input_scale
        design = np.concatenate([normalized, np.ones((len(inputs), 1), dtype=np.float64)], axis=1)
        features = design @ self.coefficients
        return EncoderOutput(
            features=features,
            point_lineage=PointLineage.from_indices(np.arange(len(inputs))),
            sample_offsets=batch.sample_offsets,
        )
