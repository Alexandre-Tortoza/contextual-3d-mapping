"""Contracts públicos de saída: embedding por ponto e lote de embeddings.

Issue: #1.

``PointEmbedding`` é deliberadamente por-ponto e sem framework: ele não sabe
qual backbone o produziu (isso é proveniência de checkpoint, #25) e não
assume que todo ponto de entrada teve um vetor recuperável — pontos
mesclados durante discretização no encoder (#21) aparecem como entradas
``valid=False`` em vez de desaparecerem, preservando a ordenação canônica.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from point_representation.domain.csr import validate_csr_offsets


# Representa o embedding aprendido de um único ponto, mais a geometria e a
# validade que o tornam interpretável sem o batch inteiro em mãos.
@dataclass(frozen=True)
class PointEmbedding:
    """O embedding aprendido de um ponto, com sua geometria e validade.

    Argumentos:
        coordinates: coordenadas XYZ do ponto que este embedding representa.
        vector: o vetor de embedding; ignorado quando ``valid`` é ``False``.
        dimension: dimensão declarada de ``vector``.
        valid: ``False`` quando o encoder não conseguiu produzir um embedding
            específico para este ponto (ex: mesclado com outro na
            discretização interna do backbone, #21).
        confidence: confiança opcional do encoder para este embedding, em
            ``[0, 1]``, quando o backbone concreto expõe uma.
    """

    coordinates: tuple[float, float, float]
    vector: tuple[float, ...]
    dimension: int
    valid: bool = True
    confidence: float | None = None

    # Valida a geometria e, apenas quando o embedding é válido, o vetor e sua
    # dimensão declarada — um embedding inválido não precisa carregar um
    # vetor coerente, já que ele não deve ser consumido como similaridade.
    def __post_init__(self) -> None:
        """Valida coordenadas e, quando ``valid``, o vetor e sua dimensão."""
        if len(self.coordinates) != 3 or not all(math.isfinite(value) for value in self.coordinates):
            raise ValueError(f"PointEmbedding.coordinates must be 3 finite values, got {self.coordinates!r}.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"PointEmbedding.confidence must be in [0, 1], got {self.confidence}.")
        if self.valid:
            if not self.vector:
                raise ValueError("PointEmbedding.vector must not be empty when valid=True.")
            if len(self.vector) != self.dimension:
                raise ValueError(
                    f"PointEmbedding dimension mismatch: declared {self.dimension}, "
                    f"got vector of length {len(self.vector)}."
                )
            if any(math.isnan(component) or math.isinf(component) for component in self.vector):
                raise ValueError("PointEmbedding.vector must be finite (no NaN/Inf).")


# Representa um lote de embeddings de múltiplas amostras preservando as
# fronteiras entre amostras, para que um consumidor (#49) recupere a saída de
# cada amostra de origem sem recontar pontos.
@dataclass(frozen=True)
class BatchedPointEmbeddings:
    """Embeddings de múltiplas amostras, concatenados em ordem canônica.

    Argumentos:
        embeddings: todos os :class:`PointEmbedding` do lote, em ordem, com
            os de cada amostra contíguos.
        sample_offsets: offsets estilo CSR; a amostra ``i`` ocupa
            ``embeddings[sample_offsets[i]:sample_offsets[i+1]]``. Tem
            ``num_samples + 1`` entradas, começando em ``0`` e terminando em
            ``len(embeddings)``.
    """

    embeddings: tuple[PointEmbedding, ...]
    sample_offsets: tuple[int, ...]

    # Valida que os offsets descrevem uma partição válida e contígua de
    # ``embeddings`` — um offset incoerente corromperia silenciosamente a
    # atribuição de embeddings à amostra de origem.
    def __post_init__(self) -> None:
        """Valida que ``sample_offsets`` particiona ``embeddings`` corretamente."""
        validate_csr_offsets(
            self.sample_offsets, len(self.embeddings), field="BatchedPointEmbeddings.sample_offsets"
        )

    # Divide o lote de volta em uma tupla por amostra, na mesma ordem de
    # ``sample_offsets``. Existe para que consumidores (#49) nunca precisem
    # fatiar ``embeddings`` manualmente.
    def per_sample(self) -> tuple[tuple[PointEmbedding, ...], ...]:
        """Retorna os embeddings agrupados por amostra, na ordem do batch."""
        return tuple(
            self.embeddings[start:end]
            for start, end in zip(self.sample_offsets, self.sample_offsets[1:], strict=False)
        )
