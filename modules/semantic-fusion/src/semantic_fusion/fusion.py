"""Fusão multi-observação de claims semânticos ancorados em um ponto do mapa."""

from __future__ import annotations

from collections.abc import Iterable

from .models import FusedPointContext, SemanticContribution


# Funde as propostas concorrentes de um ponto do mapa em um contexto único.
# Existe porque, a partir de múltiplos keyframes, o mesmo ponto persistido é
# classificado por várias observações: sem uma regra explícita, a última
# gravação venceria por acidente de ordem de execução. É chamada pelo
# mapping-runtime ao compor um trecho contextual.
def fuse_point_contributions(
    contributions: Iterable[SemanticContribution],
) -> FusedPointContext:
    """Escolhe o label primário de um ponto preservando todos os contribuintes.

    A ordenação usa confiança calibrada quando existe, confiança bruta como
    alternativa, e sinais de suporte apenas como desempate. Empates restantes
    são resolvidos pelo instante da observação e pelo identificador da região,
    para que a mesma entrada produza sempre a mesma saída.

    Argumentos:
        contributions: propostas de classificação para o mesmo ponto.
    Retorna:
        contexto fundido com label primário, concordância e contribuintes.
    Levanta:
        ValueError: se nenhuma contribuição for informada.
    """
    ranked = sorted(contributions, key=lambda item: item.ranking_key)
    if not ranked:
        raise ValueError("fusion requires at least one contribution.")
    primary = ranked[0]
    agreeing = sum(1 for item in ranked if item.label == primary.label)
    return FusedPointContext(
        label=primary.label,
        observation_id=primary.observation_id,
        region_id=primary.region_id,
        confidence=(
            primary.calibrated_confidence
            if primary.calibrated_confidence is not None
            else primary.confidence
        ),
        agreement=agreeing / len(ranked),
        contributions=tuple(ranked),
    )
