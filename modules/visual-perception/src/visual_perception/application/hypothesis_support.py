"""Suporte de hipótese a partir da evidência alinhada à linguagem.

Issues: #214 (produção do sinal), #204 (o refinamento que o consome).

Até esta etapa existir, o módulo tinha **uma** fonte de semântica. O reasoner
multimodal dizia ``wall``, e nada no pipeline conseguia concordar ou discordar
disso. Os 121 vetores CLIP que o estágio de evidência produzia por frame —
três crops por região mais a cena — eram calculados, pagos em latência e VRAM,
e descartados antes de ``PipelineResult``. ``LanguageAlignedEncoder.encode_text``
existia no port, estava implementado nos dois adapters, e não era chamado por
nenhuma linha de código de produção.

Esta etapa fecha esse circuito, e o faz com um custo quase nulo: os embeddings
de imagem **já foram calculados** pelo estágio de evidência multi-contexto. O
que se acrescenta é o encoding de texto de cada conceito distinto do frame —
algumas dezenas de chamadas curtas, memoizadas por frame.

O que ela **não** faz, e por que:

- **não converte cosseno em confiança.** Medido nos frames de referência, a
  similaridade CLIP região-texto vive entre 0,13 e 0,27 e a margem entre a
  hipótese primária e a melhor alternativa tem mediana de 0,019 a 0,027. Além
  disso, a literatura mostra que esse cosseno é uma mistura enviesada de escala
  da região e de especificidade do termo (arXiv:2607.10993): regiões grandes e
  termos específicos pontuam mais alto independentemente do conteúdo. Um número
  desses não é probabilidade de acerto, e chamá-lo de ``confidence`` seria
  repetir exatamente o erro que este módulo já mediu no score do VLM;
- **não classifica em vocabulário aberto.** Como classificador sobre o
  vocabulário do frame, o alinhamento concorda com o primário do reasoner em
  apenas 6 a 13 de 40 regiões. Como **árbitro entre as hipóteses que o próprio
  produtor registrou**, concorda em 54% a 64%, discorda em 16% a 22% e não
  distingue em 14% a 27%. É a segunda pergunta que tem resposta útil, e é ela
  que esta etapa faz;
- **não decide.** O sinal é anexado à claim ao lado do score bruto. Quem decide
  o que fazer com a discordância é o refinamento (#204) e a calibração (#196).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from visual_perception.application.support import fingerprint_of
from visual_perception.config import HypothesisSupportConfig, LanguageEmbeddingConfig
from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace, LanguageEmbedding
from visual_perception.domain.region_evidence import EvidenceSlot, EvidenceState
from visual_perception.domain.regions import ObservedRegion
from visual_perception.domain.semantic_support import HypothesisSupportSignal, SupportSignalStatus
from visual_perception.domain.semantics import (
    IDENTITY_CLAIM_KINDS,
    SemanticClaim,
    normalize_claim_value,
)
from visual_perception.ports.language_embedding import LanguageAlignedEncoder

#: Nome do estágio em ``ModelProvenance`` e nos reports.
STAGE = "hypothesis_support"


# Registra a falha de um sinal que não pôde ser produzido por um motivo de
# execução (encoder indisponível, texto rejeitado). Existe separada do status
# ``unavailable`` do próprio sinal porque são coisas diferentes: o status
# descreve o que a claim recebeu, e isto descreve o que quebrou — e o chamador
# precisa poder reportar o segundo sem varrer todas as claims do frame.
@dataclass(frozen=True)
class SignalExtractionFailure:
    """Uma falha isolada ao produzir sinais de suporte para uma região ou conceito."""

    subject: str
    reason: str


# Agrupa o resultado da etapa: as regiões já com os sinais anexados, as falhas
# isoladas e quantas chamadas de encoding de texto foram feitas. A contagem de
# chamadas existe porque o custo deste estágio precisa ser auditável junto com
# o dos demais no manifest de validação.
@dataclass(frozen=True)
class HypothesisSupportResult:
    """As regiões com sinais anexados, as falhas isoladas e o custo medido."""

    regions: tuple[ObservedRegion, ...]
    failures: tuple[SignalExtractionFailure, ...] = ()
    text_encode_calls: int = 0
    measured_signals: int = 0


# Deriva a identidade do espaço em que a comparação texto-imagem acontece.
# Existe para que o sinal declare o espaço que mediu, e para que dois sinais de
# checkpoints diferentes nunca sejam agregados como se fossem o mesmo.
def language_space(config: LanguageEmbeddingConfig) -> EmbeddingSpace:
    """Retorna o :class:`EmbeddingSpace` alinhado a linguagem declarado pela configuração."""
    return EmbeddingSpace(
        model_id=config.backend,
        checkpoint=config.checkpoint,
        dimension=config.dimension,
        modality=EmbeddingModality.LANGUAGE_ALIGNED,
        normalized=config.normalize,
    )


# Ponto de entrada público da etapa: anexa a cada hipótese de identidade os
# sinais de alinhamento medidos em cada slot configurado. Chamada pelo pipeline
# canônico logo depois da interpretação de região e antes da calibração, que é
# quem consome os sinais.
def attach_hypothesis_signals(
    regions: tuple[ObservedRegion, ...],
    language_embeddings: tuple[LanguageEmbedding, ...],
    encoder: LanguageAlignedEncoder,
    config: HypothesisSupportConfig,
    embedding_config: LanguageEmbeddingConfig,
) -> HypothesisSupportResult:
    """Mede, por slot, se a evidência alinhada sustenta cada hipótese da região.

    Argumentos:
        regions: as regiões já interpretadas, com hipóteses de identidade.
        language_embeddings: os embeddings de crop já produzidos pelo estágio de
            evidência multi-contexto. Reusá-los é o que torna esta etapa barata:
            nenhuma imagem é codificada de novo.
        encoder: o encoder alinhado a linguagem, usado apenas para texto.
        config: quais slots comparar, com qual template e qual piso de margem.
        embedding_config: identifica o espaço em que a comparação acontece.
    Retorna:
        as regiões com sinais anexados, mais falhas e custo medidos.
    """
    if not config.enabled:
        return HypothesisSupportResult(regions=regions)

    vectors = {embedding.embedding_id: np.asarray(embedding.vector) for embedding in language_embeddings}
    space = language_space(embedding_config)
    slots = tuple(EvidenceSlot(name) for name in config.slots)
    encoder_state = _TextEncoder(encoder, config, embedding_config)

    updated: list[ObservedRegion] = []
    measured = 0
    for region in regions:
        region_result = _signals_for_region(region, vectors, slots, space, encoder_state, config)
        measured += region_result[1]
        updated.append(region_result[0])

    return HypothesisSupportResult(
        regions=tuple(updated),
        failures=encoder_state.failures,
        text_encode_calls=encoder_state.calls,
        measured_signals=measured,
    )


# Codifica conceitos em texto uma única vez por frame e isola a falha de um
# conceito das demais. Existe como objeto, e não como função com dict solto,
# porque três coisas precisam viajar juntas: o cache, a contagem de chamadas e
# a lista de falhas — e passá-las separadas por quatro níveis de função era o
# caminho para elas divergirem.
class _TextEncoder:
    """Cache de embeddings de texto por frame, com isolamento de falha por conceito."""

    # Guarda o encoder e as duas configurações que definem o texto enviado e o
    # espaço em que ele cai.
    def __init__(
        self,
        encoder: LanguageAlignedEncoder,
        config: HypothesisSupportConfig,
        embedding_config: LanguageEmbeddingConfig,
    ) -> None:
        """Inicializa o cache vazio para um frame."""
        self._encoder = encoder
        self._config = config
        self._embedding_config = embedding_config
        self._cache: dict[str, np.ndarray | None] = {}
        self._failures: list[SignalExtractionFailure] = []
        self.calls = 0

    # Expõe as falhas acumuladas ao chamador, como tupla imutável.
    @property
    def failures(self) -> tuple[SignalExtractionFailure, ...]:
        """As falhas de encoding observadas neste frame."""
        return tuple(self._failures)

    # Retorna o vetor de um conceito, memoizado. Um conceito que falhou fica
    # memoizado como ``None``: falhar uma vez por frame é diagnóstico, falhar
    # uma vez por região seria ruído multiplicado pelo número de regiões.
    def vector_for(self, concept: str) -> np.ndarray | None:
        """Retorna o embedding de texto de ``concept``, ou ``None`` se ele falhou."""
        if concept in self._cache:
            return self._cache[concept]
        text = self._config.prompt_template.format(concept=concept)
        try:
            self.calls += 1
            vector = np.asarray(self._encoder.encode_text(text, self._embedding_config))
        except (ValueError, TypeError, KeyError) as error:
            self._failures.append(SignalExtractionFailure(f"concept:{concept}", str(error)))
            self._cache[concept] = None
            return None
        if vector.ndim != 1 or vector.size != self._embedding_config.dimension:
            self._failures.append(
                SignalExtractionFailure(
                    f"concept:{concept}",
                    f"text embedding has shape {vector.shape}, expected "
                    f"({self._embedding_config.dimension},)",
                )
            )
            self._cache[concept] = None
            return None
        self._cache[concept] = vector
        return vector


# Produz os sinais de todas as hipóteses de uma região e devolve a região
# atualizada junto de quantos sinais foram efetivamente medidos. Helper de
# attach_hypothesis_signals, separado para que a falha de uma região não
# interrompa as demais.
def _signals_for_region(
    region: ObservedRegion,
    vectors: dict[str, np.ndarray],
    slots: tuple[EvidenceSlot, ...],
    space: EmbeddingSpace,
    encoder: _TextEncoder,
    config: HypothesisSupportConfig,
) -> tuple[ObservedRegion, int]:
    """Anexa sinais às hipóteses de identidade de uma região."""
    identity = _unmeasured_identity_claims(region, config.source)
    if not identity:
        return region, 0

    # As concorrentes são **todas** as hipóteses de identidade da região, e não
    # só as ainda não medidas: uma claim que o refinamento acrescentou compete
    # com a original, e medi-la contra um conjunto reduzido daria uma margem que
    # não corresponde ao estado do frame.
    concepts = _distinct_concepts(
        [claim for claim in region.claims if claim.kind in IDENTITY_CLAIM_KINDS]
    )
    if len(concepts) < 2:
        reason = "single hypothesis: alignment has no competing concept to arbitrate against"
        return _with_signals(region, identity, {concept: _unavailable_row(slots, config.source, reason)
                                                for concept in concepts}), 0

    scores: dict[str, dict[EvidenceSlot, float | None]] = {
        concept: dict.fromkeys(slots, None) for concept in concepts
    }
    unavailable: dict[EvidenceSlot, str] = {}
    for slot in slots:
        image_vector = _slot_vector(region, slot, vectors)
        if image_vector is None:
            unavailable[slot] = f"no language-aligned embedding available for slot {slot.value!r}"
            continue
        for concept in concepts:
            text_vector = encoder.vector_for(concept)
            if text_vector is None:
                continue
            scores[concept][slot] = float(image_vector @ text_vector)

    measured = 0
    by_concept: dict[str, tuple[HypothesisSupportSignal, ...]] = {}
    for concept in concepts:
        signals: list[HypothesisSupportSignal] = []
        for slot in slots:
            score = scores[concept][slot]
            competing = [
                other_score
                for other in concepts
                if other != concept and (other_score := scores[other][slot]) is not None
            ]
            if score is None or not competing:
                signals.append(
                    _unavailable(
                        slot,
                        config.source,
                        concept,
                        unavailable.get(slot, "the competing hypothesis could not be encoded"),
                    )
                )
                continue
            margin = score - max(competing)
            signals.append(
                HypothesisSupportSignal(
                    source=config.source,
                    hypothesis=concept,
                    slot=slot,
                    status=_status_for(margin, config.indistinguishable_margin),
                    score=score,
                    margin=margin,
                    space=space,
                )
            )
            measured += 1
        by_concept[concept] = tuple(signals)
    return _with_signals(region, identity, by_concept), measured


# Classifica uma margem contra o piso configurado. Existe como função nomeada
# porque é a única regra de decisão desta etapa, e ela precisa ser óbvia: acima
# do piso a favor, acima do piso contra, e no meio o empate explícito que
# impede a etapa de inventar um vencedor.
def _status_for(margin: float, floor: float) -> SupportSignalStatus:
    """Converte uma margem em status, com empate explícito abaixo do piso."""
    if abs(margin) < floor:
        return SupportSignalStatus.INDISTINGUISHABLE
    return SupportSignalStatus.SUPPORTS if margin > 0.0 else SupportSignalStatus.CONTRADICTS


# Lista os conceitos distintos entre as hipóteses de identidade da região,
# preservando a ordem de aparição. A comparação é a mesma normalização trivial
# usada em todo o módulo: caixa e espaçamento, nunca ontologia.
def _distinct_concepts(claims: list[SemanticClaim]) -> tuple[str, ...]:
    """Retorna os valores normalizados distintos das hipóteses, na ordem original."""
    seen: list[str] = []
    for claim in claims:
        value = normalize_claim_value(claim.value)
        if value not in seen:
            seen.append(value)
    return tuple(seen)


# Resolve o vetor de imagem de um slot pela referência de artifact que o
# próprio slot guarda. Existe para que esta etapa não reimplemente a convenção
# de nomes de embedding: ela pergunta ao slot qual artifact o representa.
def _slot_vector(
    region: ObservedRegion, slot: EvidenceSlot, vectors: dict[str, np.ndarray]
) -> np.ndarray | None:
    """Retorna o embedding de imagem do slot, ou ``None`` quando indisponível."""
    evidence = region.evidence_for(slot)
    if evidence is None or evidence.state is not EvidenceState.AVAILABLE:
        return None
    if evidence.artifact_ref is None:
        return None
    return vectors.get(evidence.artifact_ref)


# Constrói a linha de sinais indisponíveis de um conceito, um por slot.
def _unavailable_row(
    slots: tuple[EvidenceSlot, ...], source: str, reason: str
) -> tuple[HypothesisSupportSignal, ...]:
    """Retorna um sinal ``unavailable`` por slot, com o mesmo motivo."""
    return tuple(_unavailable(slot, source, "", reason) for slot in slots)


# Constrói um sinal indisponível. O ``hypothesis`` vazio é preenchido pelo
# chamador que conhece a claim, porque o contract exige que o sinal nomeie a
# hipótese sob a qual está arquivado.
def _unavailable(
    slot: EvidenceSlot, source: str, hypothesis: str, reason: str
) -> HypothesisSupportSignal:
    """Retorna um sinal ``unavailable`` com o motivo preservado."""
    return HypothesisSupportSignal(
        source=source,
        hypothesis=hypothesis or "unknown",
        slot=slot,
        status=SupportSignalStatus.UNAVAILABLE,
        reason=reason,
    )


# Anexa os sinais às claims de identidade da região, preservando tudo o mais.
# A geometria não é tocada aqui: esta etapa só acrescenta evidência a claims.
def _with_signals(
    region: ObservedRegion,
    identity: list[SemanticClaim],
    by_concept: dict[str, tuple[HypothesisSupportSignal, ...]],
) -> ObservedRegion:
    """Devolve a região com os sinais anexados a cada hipótese de identidade."""
    identity_ids = {id(claim) for claim in identity}
    claims: list[SemanticClaim] = []
    for claim in region.claims:
        if id(claim) not in identity_ids:
            claims.append(claim)
            continue
        concept = normalize_claim_value(claim.value)
        signals = tuple(
            dataclasses.replace(signal, hypothesis=claim.value)
            for signal in by_concept.get(concept, ())
        )
        claims.append(dataclasses.replace(claim, signals=claim.signals + signals))
    return dataclasses.replace(region, claims=tuple(claims))


# Lista as hipóteses de identidade que ainda não foram medidas por esta fonte.
# Existe para que a etapa seja **idempotente**: ela roda uma vez depois da
# interpretação e de novo depois do refinamento, e um segundo passe deve medir
# só as claims que o refinamento acrescentou. Remedir as antigas violaria o
# contract (um sinal por fonte e slot) e, pior, apagaria o registro do que se
# sabia quando aquela hipótese foi afirmada.
def _unmeasured_identity_claims(region: ObservedRegion, source: str) -> list[SemanticClaim]:
    """Retorna as claims de identidade que ainda não têm sinal desta fonte."""
    return [
        claim
        for claim in region.claims
        if claim.kind in IDENTITY_CLAIM_KINDS
        and not any(signal.source == source for signal in claim.signals)
    ]


# Produz o fingerprint desta etapa, usado na proveniência de qualquer claim
# derivada dos seus sinais. Existe para que uma mudança de template ou de piso
# de margem seja rastreável até o resultado que ela produziu.
def support_fingerprint(config: HypothesisSupportConfig) -> str:
    """Retorna o fingerprint estável da configuração de suporte de hipótese."""
    return fingerprint_of(config)
