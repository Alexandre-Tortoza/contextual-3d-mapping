"""Contract de suporte semântico estruturado e estado de calibração.

Issue: #195.

Uma confiança solta em uma claim não diz de onde ela veio nem se foi
verificada. Este contract a substitui por um registro estruturado que
mantém separados:

- ``source_reliability``: quão confiável é o produtor daquele tipo de claim;
- ``visual_support``: quanta evidência visual independente sustenta a claim;
- ``region_quality``: qualidade da geometria da região que originou o crop;
- ``contradiction_support``: quanta evidência aponta no sentido contrário.

O campo ``state`` torna impossível confundir um score bruto com um score
calibrado: apenas ``calibrated`` carrega ``calibrated_confidence``, e
``abstained``/``failed`` exigem um motivo textual. Nenhum estado converte
ausência de evidência em número.

Treinar o modelo de calibração e fundir posteriores multi-view ficam fora
deste contract (ver #196 para a aplicação da regra de calibração).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from visual_perception.domain.confidence import ConfidenceScore

#: Versões de artifact de calibração que este módulo sabe interpretar.
#: Uma versão fora deste conjunto falha na validação em vez de ser aceita
#: como se fosse compatível (critério de aceitação da #195).
SUPPORTED_CALIBRATION_VERSIONS = frozenset({"calibration/1"})


# Enumera os estados possíveis do suporte de uma claim. Existe para que
# "não pontuado", "bruto", "calibrado", "abstenção" e "falha de calibração"
# sejam cinco situações distinguíveis, e não variações de um número.
class SupportState(StrEnum):
    """O estado de calibração do suporte de uma claim."""

    UNKNOWN = "unknown"
    RAW = "raw"
    CALIBRATED = "calibrated"
    ABSTAINED = "abstained"
    FAILED = "failed"


# Distingue evidência a favor de evidência contra. Existe porque a #195
# exige que múltiplas evidências de suporte *e* de contradição coexistam
# em uma mesma claim, em vez de se cancelarem em um único número.
class SupportPolarity(StrEnum):
    """Se uma evidência sustenta ou contradiz a claim."""

    SUPPORTIVE = "supportive"
    CONTRADICTORY = "contradictory"


# Referencia uma peça concreta de evidência que sustenta ou contradiz uma
# claim, amarrando-a à região e ao slot de evidência de onde veio. Existe
# para que a linhagem de uma claim calibrada seja auditável até a imagem.
@dataclass(frozen=True)
class SupportEvidence:
    """Um ponteiro para evidência que sustenta ou contradiz uma claim."""

    artifact_uri: str
    source: str
    polarity: SupportPolarity
    region_id: str | None = None
    slot: str | None = None

    # Exige uri e fonte, já que uma evidência que não pode ser localizada
    # nem atribuída a um produtor não é auditável.
    def __post_init__(self) -> None:
        """Rejeita evidência sem uri ou sem produtor identificado."""
        if not self.artifact_uri:
            raise ValueError("SupportEvidence.artifact_uri must not be empty.")
        if not self.source:
            raise ValueError("SupportEvidence.source must not be empty.")


# Agrupa os sinais de entrada que a fronteira de calibração (#196) consome
# para decidir se pontua ou se abstém. Existe separado de SemanticSupport
# porque é a *entrada* da regra, enquanto SemanticSupport é a saída dela.
@dataclass(frozen=True)
class SupportInputs:
    """Os sinais estruturados que alimentam a regra de calibração de uma claim."""

    claim_kind: str
    source: str
    raw_confidence: ConfidenceScore | None = None
    source_reliability: float | None = None
    visual_support: float | None = None
    region_quality: float | None = None
    contradiction_support: float | None = None
    domain: str | None = None
    evidence: tuple[SupportEvidence, ...] = field(default_factory=tuple)

    # Valida que todo sinal escalar informado é uma fração em [0, 1],
    # falhando cedo em vez de deixar um sinal fora de escala distorcer a
    # calibração.
    def __post_init__(self) -> None:
        """Rejeita sinais escalares fora de ``[0, 1]`` e claim/source vazios."""
        if not self.claim_kind:
            raise ValueError("SupportInputs.claim_kind must not be empty.")
        if not self.source:
            raise ValueError("SupportInputs.source must not be empty.")
        _validate_signals(self)


# Representa o suporte estruturado de uma claim e o estado da sua
# calibração. Existe como o substituto auditável de um número de confiança
# solto: cada claim passa a declarar de onde sua certeza vem, ou por que
# ela não existe.
@dataclass(frozen=True)
class SemanticSupport:
    """O suporte estruturado de uma claim e o estado da sua calibração.

    ``calibrated_confidence`` só existe no estado ``calibrated``. Nos
    demais estados ele é obrigatoriamente ``None``, para que um consumidor
    nunca confunda um score bruto com um calibrado.
    """

    state: SupportState
    source_reliability: float | None = None
    visual_support: float | None = None
    region_quality: float | None = None
    contradiction_support: float | None = None
    calibrated_confidence: ConfidenceScore | None = None
    evidence: tuple[SupportEvidence, ...] = field(default_factory=tuple)
    domain: str | None = None
    reason: str | None = None
    calibration_version: str | None = None
    calibration_artifact: str | None = None

    # Impõe as invariantes que separam bruto de calibrado: só o estado
    # calibrado carrega score, e só ele exige uma versão de calibração
    # suportada; abstenção e falha exigem um motivo legível.
    def __post_init__(self) -> None:
        """Valida a coerência entre estado, score calibrado e proveniência."""
        _validate_signals(self)
        if self.state is SupportState.CALIBRATED:
            if self.calibrated_confidence is None:
                raise ValueError("A calibrated SemanticSupport must carry a calibrated_confidence.")
            if self.calibration_version is None:
                raise ValueError("A calibrated SemanticSupport must record its calibration_version.")
        elif self.calibrated_confidence is not None:
            raise ValueError(
                f"A {self.state.value!r} SemanticSupport must not carry a calibrated_confidence."
            )
        if self.state in (SupportState.ABSTAINED, SupportState.FAILED) and not self.reason:
            raise ValueError(f"A {self.state.value!r} SemanticSupport must explain itself with a reason.")
        if self.calibration_version is not None and self.calibration_version not in (
            SUPPORTED_CALIBRATION_VERSIONS
        ):
            raise ValueError(
                f"Unsupported calibration_version {self.calibration_version!r}; "
                f"expected one of {sorted(SUPPORTED_CALIBRATION_VERSIONS)}."
            )

    # Responde se este suporte produziu um score utilizável. Usada por quem
    # ordena ou filtra claims calibradas sem inspecionar o estado na mão.
    @property
    def is_calibrated(self) -> bool:
        """Indica se há um score calibrado válido disponível."""
        return self.state is SupportState.CALIBRATED

    # Filtra as evidências de uma polaridade. Existe para que os reports de
    # avaliação (#199) contem apoio e contradição separadamente sem
    # replicar o filtro.
    def evidence_with(self, polarity: SupportPolarity) -> tuple[SupportEvidence, ...]:
        """Retorna as evidências de ``polarity``, preservando a ordem."""
        return tuple(item for item in self.evidence if item.polarity is polarity)


# Valida em um só lugar que todo sinal escalar de suporte é uma fração
# válida. Existe porque SupportInputs e SemanticSupport compartilham
# exatamente os mesmos quatro campos e a mesma regra.
def _validate_signals(holder: SupportInputs | SemanticSupport) -> None:
    """Rejeita sinais de suporte que não sejam frações reais em ``[0, 1]``."""
    for name in ("source_reliability", "visual_support", "region_quality", "contradiction_support"):
        value = getattr(holder, name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a real number or None, got {value!r}.")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")


# Converte um SemanticSupport (ou a sua ausência) em um dict serializável.
# Usada por infrastructure/serialization.py ao gravar uma SemanticClaim.
def semantic_support_to_dict(support: SemanticSupport | None) -> dict[str, Any] | None:
    """Converte um :class:`SemanticSupport` em um dict serializável, preservando ``None``."""
    if support is None:
        return None
    calibrated = support.calibrated_confidence
    return {
        "state": support.state.value,
        "source_reliability": support.source_reliability,
        "visual_support": support.visual_support,
        "region_quality": support.region_quality,
        "contradiction_support": support.contradiction_support,
        "calibrated_confidence": (
            None if calibrated is None else {"value": calibrated.value, "source": calibrated.source}
        ),
        "evidence": [
            {
                "artifact_uri": item.artifact_uri,
                "source": item.source,
                "polarity": item.polarity.value,
                "region_id": item.region_id,
                "slot": item.slot,
            }
            for item in support.evidence
        ],
        "domain": support.domain,
        "reason": support.reason,
        "calibration_version": support.calibration_version,
        "calibration_artifact": support.calibration_artifact,
    }


# Reconstrói um SemanticSupport a partir do dict — lado inverso de
# semantic_support_to_dict, revalidando as invariantes em vez de confiar no
# que estava gravado.
def semantic_support_from_dict(payload: dict[str, Any] | None) -> SemanticSupport | None:
    """Reconstrói um :class:`SemanticSupport` validado, preservando ``None``."""
    if payload is None:
        return None
    calibrated = payload.get("calibrated_confidence")
    return SemanticSupport(
        state=SupportState(payload["state"]),
        source_reliability=payload.get("source_reliability"),
        visual_support=payload.get("visual_support"),
        region_quality=payload.get("region_quality"),
        contradiction_support=payload.get("contradiction_support"),
        calibrated_confidence=None if calibrated is None else ConfidenceScore(**calibrated),
        evidence=tuple(
            SupportEvidence(
                artifact_uri=item["artifact_uri"],
                source=item["source"],
                polarity=SupportPolarity(item["polarity"]),
                region_id=item.get("region_id"),
                slot=item.get("slot"),
            )
            for item in payload.get("evidence", ())
        ),
        domain=payload.get("domain"),
        reason=payload.get("reason"),
        calibration_version=payload.get("calibration_version"),
        calibration_artifact=payload.get("calibration_artifact"),
    )
