"""Contract de suporte semântico estruturado e estado de calibração.

Issues: #195 (suporte calibrado), #214 (sinal de suporte independente).

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

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from visual_perception.domain.confidence import ConfidenceScore
from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace
from visual_perception.domain.region_evidence import EvidenceSlot

#: Versões de artifact de calibração que este módulo sabe interpretar.
#: Uma versão fora deste conjunto falha na validação em vez de ser aceita
#: como se fosse compatível (critério de aceitação da #195).
SUPPORTED_CALIBRATION_VERSIONS = frozenset({"calibration/1"})


# Enumera o que um sinal independente diz sobre uma hipótese. Existe porque a
# medição que motivou este contract encontrou **três** desfechos com massa real,
# e não dois: em 13 a 25 de 92 regiões dos frames de referência, a margem entre
# a hipótese primária e a melhor alternativa fica abaixo do ruído do próprio
# sinal. Colapsar esse terceiro caso em "discorda" inventaria um veredito que a
# medida não sustenta (ver docs/visual-context-sota-review.md, §3.1).
class SupportSignalStatus(StrEnum):
    """O que um sinal independente afirma sobre uma hipótese de identidade."""

    #: O sinal pontua esta hipótese acima da melhor concorrente, por uma
    #: margem maior que o piso configurado.
    SUPPORTS = "supports"
    #: O sinal pontua uma concorrente acima desta hipótese.
    CONTRADICTS = "contradicts"
    #: A margem entre esta hipótese e a melhor concorrente está abaixo do
    #: piso: o sinal existe, mas não distingue as duas.
    INDISTINGUISHABLE = "indistinguishable"
    #: O sinal não pôde ser produzido (evidência ausente, espaço incompatível,
    #: ou nenhuma hipótese concorrente contra a qual comparar).
    UNAVAILABLE = "unavailable"


# Registra uma medida independente sobre **uma** hipótese de identidade, feita
# em **um** slot de evidência. Existe porque a única alternativa disponível era
# transformar um cosseno em ``confidence``, e isso está medido como errado
# duas vezes: nos nossos frames a margem mediana é de 0,019 a 0,027 num intervalo
# de similaridade que vive entre 0,13 e 0,27, e a literatura mostra que o cosseno
# região-texto é uma mistura enviesada de escala e especificidade semântica
# (arXiv:2607.10993). O score bruto fica aqui, nomeado como o que é; a decisão
# fica com a calibração, que consome estes sinais.
@dataclass(frozen=True)
class HypothesisSupportSignal:
    """Uma medida independente a favor ou contra uma hipótese de identidade.

    ``score`` é a similaridade bruta no espaço declarado por ``space``, nunca
    uma probabilidade. ``margin`` é ``score`` menos o score da melhor hipótese
    concorrente da mesma região: é a grandeza comparável, e é ela que decide o
    ``status`` contra o piso configurado.
    """

    source: str
    hypothesis: str
    slot: EvidenceSlot
    status: SupportSignalStatus
    score: float | None = None
    margin: float | None = None
    space: EmbeddingSpace | None = None
    reason: str | None = None

    # Impõe que um sinal disponível traga medida e espaço, e que um sinal
    # indisponível traga motivo e nenhuma medida: sem isso, um sinal sem
    # evidência ficaria indistinguível de um sinal neutro.
    def __post_init__(self) -> None:
        """Valida a coerência entre status, medida, espaço e motivo."""
        if not self.source:
            raise ValueError("HypothesisSupportSignal.source must not be empty.")
        if not self.hypothesis:
            raise ValueError("HypothesisSupportSignal.hypothesis must not be empty.")
        if self.status is SupportSignalStatus.UNAVAILABLE:
            if not self.reason:
                raise ValueError("An unavailable HypothesisSupportSignal must explain itself.")
            if self.score is not None or self.margin is not None:
                raise ValueError("An unavailable HypothesisSupportSignal must not carry a measurement.")
            return
        if self.score is None or self.margin is None:
            raise ValueError(
                f"A {self.status.value!r} HypothesisSupportSignal requires both score and margin."
            )
        if self.space is None:
            raise ValueError(
                f"A {self.status.value!r} HypothesisSupportSignal must declare the space it measured in."
            )
        for name in ("score", "margin"):
            value = float(getattr(self, name))
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"HypothesisSupportSignal.{name} must be finite, got {value!r}.")
        if self.space.normalized and not -1.0 <= self.score <= 1.0:
            raise ValueError(
                "A signal measured in a normalized space must have a cosine score in [-1, 1], "
                f"got {self.score}."
            )
        if self.space.modality is not EmbeddingModality.LANGUAGE_ALIGNED:
            raise ValueError(
                "A hypothesis support signal compares a text hypothesis against region evidence and "
                f"therefore requires a language-aligned space, got {self.space.modality.value!r}."
            )

    # Responde se este sinal produziu uma medida utilizável. Usada por quem
    # agrega sinais sem inspecionar o status na mão.
    @property
    def is_measured(self) -> bool:
        """Indica que o sinal foi efetivamente medido, qualquer que seja o desfecho."""
        return self.status is not SupportSignalStatus.UNAVAILABLE


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


# Converte um HypothesisSupportSignal em um dict serializável. Usada por
# infrastructure/serialization.py ao gravar uma SemanticClaim, e pelos reports
# de avaliação que precisam contar desfechos por slot.
def support_signal_to_dict(signal: HypothesisSupportSignal) -> dict[str, Any]:
    """Converte um :class:`HypothesisSupportSignal` em um dict serializável."""
    return {
        "source": signal.source,
        "hypothesis": signal.hypothesis,
        "slot": signal.slot.value,
        "status": signal.status.value,
        "score": signal.score,
        "margin": signal.margin,
        "space": None if signal.space is None else _space_to_dict(signal.space),
        "reason": signal.reason,
    }


# Reconstrói um HypothesisSupportSignal a partir do dict, revalidando as
# invariantes em vez de confiar no que estava gravado.
def support_signal_from_dict(payload: dict[str, Any]) -> HypothesisSupportSignal:
    """Reconstrói um :class:`HypothesisSupportSignal` validado a partir de um dict."""
    space = payload.get("space")
    return HypothesisSupportSignal(
        source=payload["source"],
        hypothesis=payload["hypothesis"],
        slot=EvidenceSlot(payload["slot"]),
        status=SupportSignalStatus(payload["status"]),
        score=payload.get("score"),
        margin=payload.get("margin"),
        space=None if space is None else _space_from_dict(space),
        reason=payload.get("reason"),
    )


# Achata um EmbeddingSpace preservando a modalidade como string. Helper
# compartilhado por support_signal_to_dict; espelha o helper equivalente de
# ``domain/region_evidence.py``, que serializa o espaço do outro lado do
# contract de evidência.
def _space_to_dict(space: EmbeddingSpace) -> dict[str, Any]:
    """Converte um :class:`EmbeddingSpace` em um dict serializável."""
    return {
        "model_id": space.model_id,
        "checkpoint": space.checkpoint,
        "dimension": space.dimension,
        "modality": space.modality.value,
        "normalized": space.normalized,
    }


# Reconstrói um EmbeddingSpace validado a partir do dict. Helper de
# support_signal_from_dict.
def _space_from_dict(payload: dict[str, Any]) -> EmbeddingSpace:
    """Reconstrói um :class:`EmbeddingSpace` validado a partir de um dict."""
    return EmbeddingSpace(
        model_id=payload["model_id"],
        checkpoint=payload["checkpoint"],
        dimension=payload["dimension"],
        modality=EmbeddingModality(payload["modality"]),
        normalized=payload.get("normalized", True),
    )


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
