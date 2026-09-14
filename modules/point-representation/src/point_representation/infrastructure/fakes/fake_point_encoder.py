"""Encoder fake, determinístico e sem GPU, para testes de contract do port.

Issue: #19.

Existe para que o contract de `PointEncoder` (e qualquer composição de
treino/inferência que dependa dele) seja exercitável sem carregar um
backbone real ou uma GPU — a mesma razão pela qual `visual-perception` mantém
fakes completos para cada port seu.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from point_representation.domain.encoder_io import EncoderInputBatch, EncoderOutput
from point_representation.domain.errors import FeatureSelectionError
from point_representation.domain.lineage import PointLineage


# Projeta coordenadas e features selecionadas por uma matriz fixa e
# determinística (semeada por ``embedding_dimension``), um-para-um: nunca
# mescla pontos internamente, então seu ``point_lineage`` de saída é sempre
# a identidade.
@dataclass(frozen=True)
class FakePointEncoder:
    """Encoder fake um-para-um; satisfaz o Protocol `PointEncoder` sem framework de tensor."""

    embedding_dimension: int = 8
    required_channels: frozenset[str] = field(default_factory=frozenset)

    # Rejeita um batch sem os canais declarados obrigatórios, e projeta
    # coordenadas mais features por uma matriz determinística fixa.
    def encode(self, batch: EncoderInputBatch) -> EncoderOutput:
        """Projeta ``batch`` deterministicamente, um vetor por ponto de entrada."""
        missing = self.required_channels - set(batch.selected_channels)
        if missing:
            raise FeatureSelectionError(
                f"FakePointEncoder requires channels {sorted(missing)}, missing from batch."
            )
        combined = np.concatenate([batch.coordinates, batch.features], axis=1)
        rng = np.random.default_rng(seed=self.embedding_dimension)
        projection = rng.standard_normal((combined.shape[1], self.embedding_dimension))
        features = combined @ projection
        lineage = PointLineage.from_indices(np.arange(len(batch.coordinates)))
        return EncoderOutput(features=features, point_lineage=lineage, sample_offsets=batch.sample_offsets)
