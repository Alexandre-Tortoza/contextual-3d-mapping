"""Contracts públicos da fusão semântica ancorada em geometria."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


# Valida uma medida opcional no intervalo unitário. Existe porque confiança e
# sinais de suporte chegam do pipeline visual como floats opcionais, e um valor
# fora de faixa indicaria erro de produção, não incerteza.
def _unit_interval(value: float | None, name: str) -> float | None:
    """Valida uma medida opcional em [0, 1]."""
    if value is None:
        return None
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite value in [0, 1].")
    return number


# Representa uma classificação proposta por uma observação para um ponto do
# mapa. Existe porque, com múltiplos keyframes, o mesmo ponto passa a receber
# várias propostas concorrentes, e nenhuma delas pode ser descartada sem
# registro: a fusão escolhe uma primária, não apaga as demais.
@dataclass(frozen=True)
class SemanticContribution:
    """Classificação proposta por uma observação visual para um ponto.

    Argumentos:
        observation_id: observação visual que produziu a proposta.
        timestamp_ns: instante da observação, usado como desempate determinístico.
        region_id: região visual que cobre o ponto naquela observação.
        label: claim de label bruto proposto pela região.
        confidence: confiança declarada pelo produtor, não calibrada.
        calibrated_confidence: confiança calibrada, quando o produtor a fornece.
        support_state: estado de suporte declarado para o claim.
        visual_support: fração de evidência visual favorável ao claim.
        region_quality: qualidade geométrica da máscara da região.
    """

    observation_id: str
    timestamp_ns: int
    region_id: str
    label: str
    confidence: float | None = None
    calibrated_confidence: float | None = None
    support_state: str | None = None
    visual_support: float | None = None
    region_quality: float | None = None

    # Rejeita contribuições sem identidade ou com medidas impossíveis antes que
    # elas influenciem a escolha do label primário.
    def __post_init__(self) -> None:
        """Valida identidade, label e medidas da contribuição."""
        if not self.observation_id.strip():
            raise ValueError("observation_id must not be empty.")
        if not self.region_id.strip():
            raise ValueError("region_id must not be empty.")
        if not self.label.strip():
            raise ValueError("label must not be empty.")
        for name in ("confidence", "calibrated_confidence", "visual_support", "region_quality"):
            object.__setattr__(self, name, _unit_interval(getattr(self, name), name))

    # Ordena propostas usando primeiro a confiança calibrada, quando ela existe,
    # e caindo para a confiança bruta. Os sinais de suporte só desempatam,
    # porque uma soma ponderada arbitrária deles fingiria uma calibração que
    # este pipeline ainda não tem.
    @property
    def ranking_key(self) -> tuple[float, float, float, int, str]:
        """Chave de ordenação determinística entre contribuições concorrentes."""
        effective = self.calibrated_confidence
        if effective is None:
            effective = self.confidence if self.confidence is not None else 0.0
        return (
            -effective,
            -(self.visual_support if self.visual_support is not None else 0.0),
            -(self.region_quality if self.region_quality is not None else 0.0),
            self.timestamp_ns,
            self.region_id,
        )


# Resultado da fusão para um ponto do mapa. Preserva a lista completa de
# contribuintes porque a concordância entre observações é evidência, e porque
# um label primário sem seus concorrentes não é auditável.
@dataclass(frozen=True)
class FusedPointContext:
    """Contexto semântico fundido de um ponto de geometria persistente.

    Argumentos:
        label: label primário escolhido entre as contribuições.
        observation_id: observação que sustentou o label primário.
        region_id: região que sustentou o label primário.
        confidence: confiança declarada pela contribuição primária.
        agreement: fração de contribuições que concordam com o label primário.
        contributions: todas as contribuições, na ordem de ranqueamento.
    """

    label: str
    observation_id: str
    region_id: str
    confidence: float | None
    agreement: float
    contributions: tuple[SemanticContribution, ...]

    # Expõe a quantidade de observações que enxergaram o ponto sem obrigar o
    # consumidor a contar a tupla, que é a pergunta feita pelo viewer.
    @property
    def contribution_count(self) -> int:
        """Quantidade de contribuições consideradas na fusão."""
        return len(self.contributions)
