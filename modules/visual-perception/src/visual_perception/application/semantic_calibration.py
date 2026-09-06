"""Calibração de claims semânticos e abstenção explícita.

Issue: #196 (aplica o contract da #195).

Esta etapa converte o suporte estruturado de uma claim em um score
calibrado **apenas quando a regra configurada declara que pode**. Em todos
os demais casos ela produz uma abstenção explícita, com motivo, em vez de
um número plausível. Os três motivos previstos pela #196 são:

- **suporte insuficiente**: a evidência visual ou a qualidade da região
  ficaram abaixo do limiar configurado, ou não foram medidas;
- **tipo de claim não suportado**: o artifact de calibração não cobre
  aquele par (produtor, tipo de claim);
- **evidência fora de distribuição**: o domínio configurado não é o
  domínio em que a tabela de calibração foi medida.

O score bruto e o suporte pré-calibração nunca são apagados: eles seguem em
``SemanticClaim.confidence`` e nos campos de sinal de
:class:`~visual_perception.domain.semantic_support.SemanticSupport`, para
que a avaliação (#199) possa comparar bruto e calibrado lado a lado.

Uma falha ao carregar ou interpretar o artifact de calibração não derruba a
observação: ela vira o estado ``failed``, que o auditor de qualidade
reporta com um código próprio, distinto de discordância semântica.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from visual_perception.config import CalibrationConfig
from visual_perception.domain.confidence import ConfidenceScore
from visual_perception.domain.region_evidence import EvidenceState
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantic_support import (
    SUPPORTED_CALIBRATION_VERSIONS,
    SemanticSupport,
    SupportEvidence,
    SupportInputs,
    SupportPolarity,
    SupportState,
)
from visual_perception.domain.semantics import ClaimKind, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext
from visual_perception.ports.calibration import ClaimCalibrator


# Registra uma claim que não pôde ser calibrada por falha da regra (e não
# por abstenção deliberada). Existe para que o chamador distinga "a
# calibração decidiu não pontuar" de "a calibração quebrou".
@dataclass(frozen=True)
class CalibrationFailure:
    """Uma falha da própria regra de calibração ao processar uma claim."""

    claim_kind: str
    value: str
    reason: str
    region_id: str | None = None


# Representa a tabela de calibração medida, já validada. Existe para que o
# artifact em disco seja convertido uma única vez em uma estrutura tipada,
# em vez de reinterpretado a cada claim.
@dataclass(frozen=True)
class CalibrationArtifact:
    """Uma tabela de calibração medida, com sua versão, domínio e digest."""

    version: str
    domain: str
    digest: str
    #: ``{producer: {claim_kind: ((lower, upper, calibrated), ...)}}``.
    bins: dict[str, dict[str, tuple[tuple[float, float, float], ...]]]
    #: ``{producer: {claim_kind: samples}}``, usado como confiabilidade da fonte.
    samples: dict[str, dict[str, int]]

    # Responde se a tabela cobre um par (produtor, tipo de claim). Usada
    # pelo calibrador para decidir entre pontuar e se abster por tipo não
    # suportado.
    def covers(self, source: str, claim_kind: str) -> bool:
        """Indica se o artifact tem dados medidos para ``(source, claim_kind)``."""
        return claim_kind in self.bins.get(source, {})

    # Converte um score bruto no valor calibrado medido para aquele bin.
    # Existe como a única implementação da consulta à tabela, para que o
    # calibrador não replique a busca binária de faixa.
    def calibrated_value(self, source: str, claim_kind: str, raw: float) -> float | None:
        """Retorna o valor calibrado do bin que contém ``raw``, ou ``None`` fora de cobertura."""
        for lower, upper, calibrated in self.bins.get(source, {}).get(claim_kind, ()):
            if lower <= raw <= upper:
                return calibrated
        return None

    # Deriva a confiabilidade da fonte a partir do volume de amostras
    # medidas. Existe para que ``source_reliability`` seja um número
    # observado (quanta evidência sustenta a tabela) e não uma constante.
    def source_reliability(self, source: str, claim_kind: str) -> float | None:
        """Retorna a confiabilidade medida da fonte, saturando em 100 amostras."""
        count = self.samples.get(source, {}).get(claim_kind)
        if count is None:
            return None
        return min(1.0, count / 100.0)


# Calibrador que aplica a tabela de confiabilidade medida, respeitando os
# limiares de abstenção da configuração. É a implementação de referência do
# port ClaimCalibrator, usada quando ``calibration.enabled`` é verdadeiro.
class ReliabilityTableCalibrator:
    """Pontua uma claim pelo bin medido correspondente, ou se abstém com motivo."""

    # Guarda o artifact validado e a configuração que define os limiares de
    # abstenção e o domínio declarado da execução.
    def __init__(self, artifact: CalibrationArtifact, config: CalibrationConfig) -> None:
        """Inicializa o calibrador com uma tabela já validada."""
        self._artifact = artifact
        self._config = config

    # Expõe a versão do contract de calibração aplicada, exigida pelo port.
    @property
    def version(self) -> str:
        """A versão declarada pelo artifact de calibração."""
        return self._artifact.version

    # Expõe o digest do artifact, para que a proveniência do resultado
    # aponte para os dados exatos usados.
    @property
    def artifact_id(self) -> str | None:
        """O digest SHA-256 do artifact de calibração."""
        return self._artifact.digest

    # Aplica a política de abstenção e, quando ela permite, converte o score
    # bruto no valor calibrado medido. Ponto de entrada do port.
    def calibrate(self, inputs: SupportInputs) -> SemanticSupport:
        """Decide entre pontuar a claim ou se abster, sempre com justificativa."""
        reason = self._abstention_reason(inputs)
        if reason is not None:
            return self._support(inputs, state=SupportState.ABSTAINED, reason=reason)

        raw = inputs.raw_confidence
        if raw is None:  # pragma: no cover - _abstention_reason já garante um score
            return self._support(
                inputs, state=SupportState.ABSTAINED, reason="unscored source: nothing to calibrate"
            )
        calibrated = self._artifact.calibrated_value(inputs.source, inputs.claim_kind, raw.value)
        if calibrated is None:
            return self._support(
                inputs,
                state=SupportState.ABSTAINED,
                reason=(
                    f"raw score {raw.value:.3f} falls outside the measured bins for "
                    f"{inputs.source!r}/{inputs.claim_kind!r}"
                ),
            )
        return self._support(
            inputs,
            state=SupportState.CALIBRATED,
            calibrated=ConfidenceScore(calibrated, source=f"calibrated:{inputs.source}"),
        )

    # Monta o SemanticSupport preservando todos os sinais medidos e a
    # proveniência do artifact, variando apenas o desfecho da decisão.
    # Existe para que pontuar e abster compartilhem exatamente a mesma
    # linhagem, sem repetir a lista de campos em cada retorno.
    def _support(
        self,
        inputs: SupportInputs,
        *,
        state: SupportState,
        reason: str | None = None,
        calibrated: ConfidenceScore | None = None,
    ) -> SemanticSupport:
        """Empacota a decisão de calibração junto dos sinais e da proveniência."""
        return SemanticSupport(
            state=state,
            source_reliability=self._artifact.source_reliability(inputs.source, inputs.claim_kind),
            visual_support=inputs.visual_support,
            region_quality=inputs.region_quality,
            contradiction_support=inputs.contradiction_support,
            calibrated_confidence=calibrated,
            evidence=inputs.evidence,
            domain=self._config.domain,
            reason=reason,
            calibration_version=self._artifact.version,
            calibration_artifact=self._artifact.digest,
        )

    # Concentra as três regras de abstenção da #196 em um único lugar
    # testável, devolvendo o motivo textual ou None quando pode pontuar.
    def _abstention_reason(self, inputs: SupportInputs) -> str | None:
        """Retorna por que esta claim não pode ser calibrada, ou ``None`` se pode."""
        if inputs.domain is not None and inputs.domain != self._artifact.domain:
            return (
                f"out-of-distribution evidence: claim domain {inputs.domain!r} was not measured by "
                f"a calibration artifact for domain {self._artifact.domain!r}"
            )
        if not self._artifact.covers(inputs.source, inputs.claim_kind):
            return (
                f"unsupported claim type: the calibration artifact has no measured data for "
                f"{inputs.source!r}/{inputs.claim_kind!r}"
            )
        if inputs.raw_confidence is None:
            return "unscored source: the producer did not report a confidence to calibrate"
        if self._config.min_visual_support > 0.0 and (
            inputs.visual_support is None or inputs.visual_support < self._config.min_visual_support
        ):
            measured = "not measured" if inputs.visual_support is None else f"{inputs.visual_support:.3f}"
            return (
                f"insufficient support: visual support {measured} is below the configured minimum "
                f"{self._config.min_visual_support:.3f}"
            )
        if self._config.min_region_quality > 0.0 and (
            inputs.region_quality is None or inputs.region_quality < self._config.min_region_quality
        ):
            measured = "not measured" if inputs.region_quality is None else f"{inputs.region_quality:.3f}"
            return (
                f"insufficient support: region quality {measured} is below the configured minimum "
                f"{self._config.min_region_quality:.3f}"
            )
        return None


# Calibrador usado quando a calibração está desligada. Existe para que o
# pipeline tenha sempre um calibrador (sem ramo condicional espalhado) e
# para deixar registrado, em cada claim, que aquele score é bruto e não
# passou por nenhuma verificação.
class UncalibratedSupport:
    """Preserva o score bruto sem jamais promovê-lo a calibrado."""

    # Guarda apenas o domínio declarado, que segue no suporte para que os
    # reports saibam em que condição a execução ocorreu.
    def __init__(self, domain: str) -> None:
        """Inicializa o caminho sem calibração para um domínio declarado."""
        self._domain = domain

    # Não há versão de calibração aplicada; o port exige o campo, e o valor
    # explicita a ausência.
    @property
    def version(self) -> str:
        """A ausência de calibração, nomeada explicitamente."""
        return "uncalibrated"

    # Não há artifact medido no caminho sem calibração.
    @property
    def artifact_id(self) -> str | None:
        """Sempre ``None``: nenhum artifact medido foi consultado."""
        return None

    # Devolve o suporte bruto (ou desconhecido, quando não houve score),
    # preservando os sinais medidos sem converter nada em certeza.
    def calibrate(self, inputs: SupportInputs) -> SemanticSupport:
        """Retorna suporte ``raw`` ou ``unknown``, nunca ``calibrated``."""
        state = SupportState.RAW if inputs.raw_confidence is not None else SupportState.UNKNOWN
        return SemanticSupport(
            state=state,
            source_reliability=inputs.source_reliability,
            visual_support=inputs.visual_support,
            region_quality=inputs.region_quality,
            contradiction_support=inputs.contradiction_support,
            evidence=inputs.evidence,
            domain=self._domain,
        )


# Calibrador usado quando o artifact configurado não pôde ser carregado.
# Existe para que uma falha de calibração seja reportada por claim e
# auditável, em vez de derrubar a observação inteira.
class FailedCalibrator:
    """Marca toda claim como falha de calibração, preservando o motivo."""

    # Guarda o motivo da falha e o domínio declarado, ambos propagados para
    # cada SemanticSupport produzido.
    def __init__(self, reason: str, domain: str) -> None:
        """Inicializa o calibrador de falha com o motivo a propagar."""
        self._reason = reason
        self._domain = domain

    # A versão continua sendo a do contract pedido, mas nenhum artifact foi
    # aplicado; o estado ``failed`` é o que comunica isso.
    @property
    def version(self) -> str:
        """A calibração que se pretendia aplicar e falhou."""
        return "failed"

    # Nenhum artifact válido foi carregado.
    @property
    def artifact_id(self) -> str | None:
        """Sempre ``None``: o artifact não pôde ser lido ou validado."""
        return None

    # Expõe o motivo da falha para quem monta o relatório de execução.
    @property
    def reason(self) -> str:
        """A explicação da falha de carregamento do artifact."""
        return self._reason

    # Produz o suporte em estado ``failed``, preservando os sinais medidos.
    def calibrate(self, inputs: SupportInputs) -> SemanticSupport:
        """Retorna suporte ``failed`` com o motivo do carregamento malsucedido."""
        return SemanticSupport(
            state=SupportState.FAILED,
            source_reliability=inputs.source_reliability,
            visual_support=inputs.visual_support,
            region_quality=inputs.region_quality,
            contradiction_support=inputs.contradiction_support,
            evidence=inputs.evidence,
            domain=self._domain,
            reason=self._reason,
        )


# Escolhe o calibrador correspondente à configuração, traduzindo qualquer
# problema de carregamento em um FailedCalibrator. Chamada pelo pipeline
# canônico uma vez por execução.
def build_calibrator(config: CalibrationConfig) -> ClaimCalibrator:
    """Constrói o calibrador da configuração, sem levantar em caso de artifact inválido.

    Argumentos:
        config: a configuração de calibração do módulo.
    Retorna:
        um :class:`~visual_perception.ports.calibration.ClaimCalibrator`.
    """
    if not config.enabled:
        return UncalibratedSupport(config.domain)
    assert config.artifact_path is not None  # imposto por CalibrationConfig.__post_init__
    try:
        artifact = load_calibration_artifact(Path(config.artifact_path))
    except (OSError, ValueError) as error:
        return FailedCalibrator(f"calibration artifact could not be loaded: {error}", config.domain)
    if artifact.version != config.version:
        return FailedCalibrator(
            f"calibration artifact declares version {artifact.version!r}, "
            f"but the configuration requests {config.version!r}",
            config.domain,
        )
    if config.method == "temperature":
        return FailedCalibrator(
            "temperature scaling has no measured artifact in this repository yet; "
            "use method='reliability_table' or supply a temperature artifact",
            config.domain,
        )
    return ReliabilityTableCalibrator(artifact, config)


# Lê e valida um artifact de calibração em disco. Existe separada de
# build_calibrator para que ferramentas de experimento possam validar um
# artifact recém-gerado sem construir um calibrador.
def load_calibration_artifact(path: Path) -> CalibrationArtifact:
    """Carrega e valida uma tabela de calibração medida.

    Argumentos:
        path: caminho do arquivo JSON versionado de calibração.
    Retorna:
        o :class:`CalibrationArtifact` validado.
    Levanta:
        OSError: se o arquivo não puder ser lido.
        ValueError: se o conteúdo não satisfizer o contract de calibração.
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON calibration artifact {path}: {error}.") from error
    if not isinstance(document, dict):
        raise ValueError(f"calibration artifact {path} must contain a JSON object.")

    version = document.get("calibration_version")
    if version not in SUPPORTED_CALIBRATION_VERSIONS:
        raise ValueError(
            f"unsupported calibration_version {version!r}; "
            f"expected one of {sorted(SUPPORTED_CALIBRATION_VERSIONS)}."
        )
    domain = document.get("domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError("calibration artifact must declare a non-empty 'domain'.")

    bins, samples = _parse_sources(document.get("sources"))
    return CalibrationArtifact(
        version=version, domain=domain, digest=digest, bins=bins, samples=samples
    )


# Converte a seção ``sources`` do artifact nos dois mapas tipados que o
# CalibrationArtifact expõe, validando bins monotônicos e valores em [0, 1].
# Helper de load_calibration_artifact.
def _parse_sources(
    raw: Any,
) -> tuple[dict[str, dict[str, tuple[tuple[float, float, float], ...]]], dict[str, dict[str, int]]]:
    """Valida e converte os bins medidos por produtor e tipo de claim."""
    if not isinstance(raw, dict) or not raw:
        raise ValueError("calibration artifact must declare a non-empty 'sources' object.")
    bins: dict[str, dict[str, tuple[tuple[float, float, float], ...]]] = {}
    samples: dict[str, dict[str, int]] = {}
    for source, kinds in raw.items():
        if not isinstance(kinds, dict) or not kinds:
            raise ValueError(f"calibration source {source!r} must map claim kinds to measured bins.")
        bins[source], samples[source] = {}, {}
        for claim_kind, entry in kinds.items():
            if claim_kind not in {kind.value for kind in ClaimKind}:
                raise ValueError(
                    f"calibration source {source!r} references unknown claim kind {claim_kind!r}."
                )
            if not isinstance(entry, dict):
                raise ValueError(f"calibration entry {source!r}/{claim_kind!r} must be an object.")
            bins[source][claim_kind] = _parse_bins(entry.get("bins"), source, claim_kind)
            count = entry.get("samples", 0)
            if type(count) is not int or count < 0:
                raise ValueError(
                    f"calibration entry {source!r}/{claim_kind!r} 'samples' must be a non-negative integer."
                )
            samples[source][claim_kind] = count
    return bins, samples


# Valida uma lista de bins ``[lower, upper, calibrated]``, exigindo faixas
# ordenadas, contíguas e dentro de [0, 1]. Helper de _parse_sources.
def _parse_bins(raw: Any, source: str, claim_kind: str) -> tuple[tuple[float, float, float], ...]:
    """Valida os bins medidos de um par (produtor, tipo de claim)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"calibration entry {source!r}/{claim_kind!r} must declare a non-empty 'bins' list.")
    parsed: list[tuple[float, float, float]] = []
    previous_upper = 0.0
    for index, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(
                f"calibration bin {index} of {source!r}/{claim_kind!r} must be [lower, upper, calibrated]."
            )
        lower, upper, calibrated = (float(value) for value in item)
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                f"calibration bin {index} of {source!r}/{claim_kind!r} must satisfy 0 <= lower < upper <= 1."
            )
        if not 0.0 <= calibrated <= 1.0:
            raise ValueError(
                f"calibration bin {index} of {source!r}/{claim_kind!r} must map into [0, 1]."
            )
        if lower < previous_upper:
            raise ValueError(
                f"calibration bins of {source!r}/{claim_kind!r} must be sorted and non-overlapping."
            )
        previous_upper = upper
        parsed.append((lower, upper, calibrated))
    return tuple(parsed)


# Deriva os sinais estruturados de suporte de uma claim a partir do que a
# observação já mediu: qualidade geométrica da região, cobertura de suporte
# denso do slot de foreground, e densidade de contradição entre claims
# irmãs. Chamada por calibrate_observation_claims para cada claim.
def derive_support_inputs(
    claim: SemanticClaim, *, region: ObservedRegion | None, siblings: tuple[SemanticClaim, ...], domain: str
) -> SupportInputs:
    """Monta os :class:`SupportInputs` de uma claim a partir de evidência já medida.

    Argumentos:
        claim: a claim a calibrar.
        region: a região dona da claim, ou ``None`` para claims de cena.
        siblings: as demais claims do mesmo kind, usadas para medir contradição.
        domain: o domínio declarado da execução.
    Retorna:
        os sinais estruturados que a regra de calibração consome.
    """
    visual_support: float | None = None
    evidence: list[SupportEvidence] = []
    region_quality: float | None = None
    if region is not None:
        region_quality = region.geometric_confidence
        for slot in region.evidence:
            if slot.state is not EvidenceState.AVAILABLE or slot.artifact_ref is None:
                continue
            if slot.is_foreground and slot.support_ratio is not None:
                visual_support = slot.support_ratio
            evidence.append(
                SupportEvidence(
                    artifact_uri=slot.artifact_ref,
                    source=claim.provenance.producer,
                    polarity=SupportPolarity.SUPPORTIVE,
                    region_id=region.region_id,
                    slot=slot.slot.value,
                )
            )
    disagreeing = tuple(other for other in siblings if other.value != claim.value)
    contradiction = len(disagreeing) / len(siblings) if siblings else 0.0
    for other in disagreeing:
        evidence.append(
            SupportEvidence(
                artifact_uri=f"claim:{other.kind.value}:{other.value}",
                source=other.provenance.producer,
                polarity=SupportPolarity.CONTRADICTORY,
                region_id=None if region is None else region.region_id,
            )
        )
    return SupportInputs(
        claim_kind=claim.kind.value,
        source=claim.provenance.producer,
        raw_confidence=claim.confidence,
        visual_support=visual_support,
        region_quality=region_quality,
        contradiction_support=contradiction,
        domain=domain,
        evidence=tuple(evidence),
    )


# Ponto de entrada público da etapa: anexa suporte calibrado (ou abstenção)
# a toda claim de região e de cena de uma observação. Chamada pelo pipeline
# canônico logo após a interpretação de região e antes da geração de
# relations.
def calibrate_observation_claims(
    regions: tuple[ObservedRegion, ...],
    scene_context: SceneContext,
    calibrator: ClaimCalibrator,
    config: CalibrationConfig,
) -> tuple[tuple[ObservedRegion, ...], SceneContext, tuple[CalibrationFailure, ...]]:
    """Anexa suporte calibrado a cada claim, preservando score bruto e evidência.

    Argumentos:
        regions: as regiões já interpretadas.
        scene_context: os claims de nível de cena.
        calibrator: a regra de calibração selecionada.
        config: a configuração de calibração (domínio e limiares).
    Retorna:
        ``(regions, scene_context, failures)`` com o suporte anexado.
    """
    failures: list[CalibrationFailure] = []
    updated_regions = tuple(
        dataclasses.replace(
            region,
            claims=_calibrate_claims(region.claims, region, calibrator, config, failures),
        )
        for region in regions
    )
    updated_scene = SceneContext(
        claims=_calibrate_claims(scene_context.claims, None, calibrator, config, failures)
    )
    return updated_regions, updated_scene, tuple(failures)


# Calibra um conjunto de claims que compartilham o mesmo dono (uma região ou
# a cena), agrupando por kind para medir contradição. Helper de
# calibrate_observation_claims.
def _calibrate_claims(
    claims: tuple[SemanticClaim, ...],
    region: ObservedRegion | None,
    calibrator: ClaimCalibrator,
    config: CalibrationConfig,
    failures: list[CalibrationFailure],
) -> tuple[SemanticClaim, ...]:
    """Anexa suporte a cada claim, medindo contradição dentro do mesmo kind."""
    by_kind: dict[ClaimKind, tuple[SemanticClaim, ...]] = {}
    for claim in claims:
        by_kind[claim.kind] = (*by_kind.get(claim.kind, ()), claim)

    calibrated: list[SemanticClaim] = []
    for claim in claims:
        siblings = tuple(other for other in by_kind[claim.kind] if other is not claim)
        inputs = derive_support_inputs(claim, region=region, siblings=siblings, domain=config.domain)
        support = calibrator.calibrate(inputs)
        if support.state is SupportState.FAILED:
            failures.append(
                CalibrationFailure(
                    claim_kind=claim.kind.value,
                    value=claim.value,
                    reason=support.reason or "calibration failed without a reason",
                    region_id=None if region is None else region.region_id,
                )
            )
        calibrated.append(dataclasses.replace(claim, support=support))
    return tuple(calibrated)
