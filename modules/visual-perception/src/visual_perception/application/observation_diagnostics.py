"""Diagnóstico estatístico de uma VisualObservation.

Responde, sem abrir nenhuma imagem, perguntas que hoje só se responde olhando
um overlay com dezenas de caixas: quantas proposals viraram quantas regiões, o
quanto a interpretação colapsou em um único label, e como as duas confianças
(semântica e geométrica) se distribuem — separadamente, porque confundi-las foi
a causa de um overlay enganoso registrado em ``docs/known-limitations.md``.

Vive em ``application/`` e não no benchmark por duas razões: é a definição do
próprio módulo sobre a qualidade da saída dele (o mesmo gênero de
``quality_audit.py``), e ``benchmarks/`` não é coberto por mypy — justamente
este código, cheio de ``float | None`` e de médias sobre conjuntos possivelmente
vazios, é o que mais se beneficia da verificação estrita.

O módulo é puro: sem ``Path``, sem ``json``, sem imagem e sem conhecimento de
rig ou dataset. Serialização e tudo que é específico do corridor-02 pertencem a
quem chama.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field

from visual_perception.domain.geometry import Mask
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.regions import (
    ObservedRegion,
    ProposalRejectionReason,
    RegionProposal,
    RejectedProposal,
    alternative_label_claims,
    primary_label_claim,
)
from visual_perception.domain.relations import RelationSource
from visual_perception.domain.semantic_support import SupportSignalStatus, SupportState
from visual_perception.domain.semantics import (
    ASSERTED_HYPOTHESIS_ROLES,
    ClaimKind,
    HypothesisRole,
    RegionKind,
    SemanticClaim,
    normalize_claim_value,
)
from visual_perception.domain.structural_consistency import StructuralVerdict, region_kind_verdict
from visual_perception.domain.visual_observation import VisualObservation

#: Acima desta fração de sobreposição com o rig, uma região final é contada
#: como sendo do ego-veículo. Espelha o default de ``ImageAreaConfig`` para que
#: medir e filtrar usem a mesma régua; o chamador pode passar a sua.
_DEFAULT_EGO_OVERLAP_THRESHOLD = 0.3

#: Abaixo desta fração dentro da área válida, uma região final é contada como
#: fora dela. Espelha o default de ``ImageAreaConfig`` pela mesma razão.
_DEFAULT_VALID_AREA_THRESHOLD = 0.5


# Descreve a distribuição de um eixo de confiança contando a ausência à parte
# dos valores. Existe porque "não pontuado" não é "pontuado com zero": o
# backend pode omitir a confiança, e somar essa ausência como 0.0 inventaria
# uma certeza baixa que ninguém afirmou.
@dataclass(frozen=True)
class ConfidenceStats:
    """Distribuição de um eixo de confiança, com a ausência contada à parte.

    Argumentos:
        count: quantos valores existem de fato.
        missing: quantos itens não têm valor nenhum.
        minimum: menor valor observado, ou ``None`` sem valores.
        maximum: maior valor observado, ou ``None`` sem valores.
        mean: média dos valores observados, ou ``None`` sem valores.
        median: mediana dos valores observados, ou ``None`` sem valores.
        distinct_values: quantos valores distintos aparecem.
        stddev: desvio padrão populacional, ou ``None`` com menos de dois valores.
        degenerate: se a distribuição não tem variância nenhuma.
    """

    count: int
    missing: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    distinct_values: int = 0
    stddev: float | None = None
    degenerate: bool = False


# Mede o quanto a interpretação de um frame colapsou em um único label. É a
# métrica que torna detectável, sem inspeção visual, o caso observado em
# corridor-02-002, onde 45 de 45 regiões saíram como "curved wall".
@dataclass(frozen=True)
class ModeCollapse:
    """Quanto a interpretação do frame colapsou em um único label.

    Argumentos:
        dominant_label: o label mais frequente, ou ``None`` sem regiões rotuladas.
        dominant_fraction: fração das regiões do frame que carregam esse label.
        distinct_labels: quantos labels primários distintos aparecem no frame.
        dominant_category: a categoria mais frequente, ou ``None`` sem categorias.
        dominant_category_fraction: fração do frame que carrega essa categoria.
        distinct_categories: quantas categorias distintas aparecem no frame.
    """

    dominant_label: str | None
    dominant_fraction: float
    distinct_labels: int
    dominant_category: str | None = None
    dominant_category_fraction: float = 0.0
    distinct_categories: int = 0


# Mede a exclusão do ego-veículo. Existe para que "a filtragem foi aplicada"
# seja uma afirmação verificável e não uma flag de linha de comando: o gate da
# #202 é que nenhuma região final represente chassi, pneus ou estrutura do rig.
@dataclass(frozen=True)
class EgoExclusionStats:
    """Quanto do rig foi descartado, e quanto dele sobrou nas regiões finais.

    Argumentos:
        proposals_rejected: proposals descartadas por sobrepor o ego-veículo.
        proposals_overlapping_ego: proposals **sobreviventes** que ainda o
            sobrepõem; o gate exige zero, e um valor diferente significa que a
            filtragem deixou passar o que deveria ter descartado.
        regions_overlapping_ego: regiões finais que ainda o sobrepõem acima do
            limiar; o gate exige zero.
        applied: se havia geometria de ego declarada nesta execução.
    """

    proposals_rejected: int = 0
    proposals_overlapping_ego: int = 0
    regions_overlapping_ego: int = 0
    applied: bool = False


# Mede a exclusão da área fora da lente. Separada de EgoExclusionStats porque
# as duas respondem perguntas diferentes e são comparadas por nome entre runs.
@dataclass(frozen=True)
class ValidAreaStats:
    """Quanto ficou fora da área útil do sensor.

    Argumentos:
        proposals_rejected: proposals descartadas por caírem fora dela.
        proposals_outside_valid_area: proposals **sobreviventes** ainda fora
            dela; o gate exige zero. É a pós-condição da filtragem, e não a
            contagem de descartes — confundir as duas faria um filtro correto
            parecer reprovado.
        regions_outside_valid_area: regiões finais majoritariamente fora dela;
            o gate exige zero.
        applied: se havia geometria de área válida declarada nesta execução.
    """

    proposals_rejected: int = 0
    proposals_outside_valid_area: int = 0
    regions_outside_valid_area: int = 0
    applied: bool = False


# Resume o que a evidência independente e a reconciliação disseram sobre o
# frame. Existe porque os números que este módulo já reportava — labels
# distintos, colapso de modo, confiança — deixaram de bastar quando o pipeline
# ganhou um segundo canal de evidência: sem estes campos, ligar ou desligar o
# suporte de hipótese não mudaria nada de mensurável no artifact.
@dataclass(frozen=True)
class ContextualDiagnostics:
    """O que os estágios de contexto produziram, contável entre runs.

    Argumentos:
        signal_statuses: histograma dos desfechos dos sinais independentes
            sobre as hipóteses primárias. Um sinal ``indistinguishable`` conta
            à parte de propósito: colapsá-lo em "sem suporte" inventaria um
            veredito que a margem medida não sustenta.
        regions_with_unsupported_primary: regiões cuja hipótese primária é
            contradita por **todos** os sinais medidos.
        regions_with_ambiguous_identity: regiões em que nenhum sinal medido
            distingue a primária das concorrentes.
        region_kind_contradictions: regiões cujo conceito e natureza declarados
            são incompatíveis, pela política determinística do domínio.
        reconciled_regions: regiões que receberam uma interpretação
            reconciliada.
        distinct_raw_labels: labels primários distintos, sem normalização. É a
            métrica contaminada, preservada para comparabilidade histórica.
        distinct_canonical_concepts: conceitos distintos depois da
            canonicalização lexical mínima. É o número que responde "quantas
            coisas diferentes este frame diz", sem contar variação de grafia.
        entity_groups: grupos de mesma superfície propostos.
        regions_in_entity_groups: regiões cobertas por algum grupo.
        supported_entity_groups: grupos cuja coerência densa os corrobora.
        semantic_relations: histograma ``(predicado, contagem)`` das relações
            inferidas por modelo.
        scene_echo_any_assertion: regiões em que **alguma** hipótese afirmada
            repete uma claim de cena. ``scene_echo_label_count`` olha só a
            primeira, e por isso não veria um eco introduzido por um passe de
            refinamento; este campo vê, e os dois coexistem para que a série
            histórica daquele continue comparável.
        competing_assertions: regiões que carregam mais de uma hipótese
            afirmada, tipicamente porque o refinamento mudou de ideia.
        geometric_relations: quantas relações vieram do caminho geométrico.
        abstained_claims: claims cuja calibração se absteve explicitamente.
        unscored_claims: claims de identidade que o produtor não pontuou.
    """

    signal_statuses: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    regions_with_unsupported_primary: int = 0
    regions_with_ambiguous_identity: int = 0
    region_kind_contradictions: int = 0
    reconciled_regions: int = 0
    distinct_raw_labels: int = 0
    distinct_canonical_concepts: int = 0
    entity_groups: int = 0
    regions_in_entity_groups: int = 0
    supported_entity_groups: int = 0
    semantic_relations: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    geometric_relations: int = 0
    abstained_claims: int = 0
    unscored_claims: int = 0
    scene_echo_any_assertion: int = 0
    competing_assertions: int = 0


# Agrega tudo que se pode afirmar sobre uma observação sem reexecutar modelo
# nenhum. Consumido pelo harness de validação, que o serializa junto dos
# artifacts do frame.
@dataclass(frozen=True)
class ObservationDiagnostics:
    """Resumo auditável de um frame, do estágio de proposals até os labels.

    Argumentos:
        region_count: regiões na observação final.
        proposal_count: proposals que discovery produziu, antes do merge.
        merged_proposal_count: proposals efetivamente atribuídas a alguma região.
        regions_with_multiple_proposals: regiões nascidas de mais de uma proposal.
        label_counts: histograma ``(label, contagem)`` do label primário por região.
        category_counts: histograma ``(category, contagem)`` por região.
        region_kinds: histograma ``(RegionKind, contagem)`` por região; uma
            região sem kind declarado conta como ``unknown``.
        semantic_confidence: distribuição da confiança dos labels primários.
        geometric_confidence: distribuição da confiança de máscara/box.
        duplicate_label_hypotheses: regiões que persistem hipóteses de label
            repetidas entre si.
        mode_collapse: o quanto o frame colapsou em um label ou categoria só.
        ego: exclusão do ego-veículo.
        fisheye: exclusão da área fora da lente.
        rejected_proposals: histograma ``(motivo, contagem)`` dos descartes.
        scene_echo_label_count: regiões cujo label repete uma claim de cena.
    """

    region_count: int
    proposal_count: int
    merged_proposal_count: int
    regions_with_multiple_proposals: int
    label_counts: tuple[tuple[str, int], ...]
    category_counts: tuple[tuple[str, int], ...]
    region_kinds: tuple[tuple[str, int], ...]
    semantic_confidence: ConfidenceStats
    geometric_confidence: ConfidenceStats
    duplicate_label_hypotheses: int
    mode_collapse: ModeCollapse
    scene_echo_label_count: int
    ego: EgoExclusionStats = field(default_factory=EgoExclusionStats)
    fisheye: ValidAreaStats = field(default_factory=ValidAreaStats)
    rejected_proposals: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    contextual: ContextualDiagnostics = field(default_factory=ContextualDiagnostics)


# Resume uma lista de valores opcionais em ConfidenceStats. Recebe os ausentes
# como None em vez de já filtrados para que a contagem de ausência seja feita
# aqui, num lugar só, e não em cada chamador.
def _summarize(values: tuple[float | None, ...]) -> ConfidenceStats:
    """Resume valores opcionais, contando os ausentes separadamente."""
    present = sorted(value for value in values if value is not None)
    missing = len(values) - len(present)
    if not present:
        return ConfidenceStats(count=0, missing=missing, minimum=None, maximum=None, mean=None, median=None)
    middle = len(present) // 2
    median = present[middle] if len(present) % 2 else (present[middle - 1] + present[middle]) / 2
    distinct = len(set(present))
    # O desvio padrão é populacional porque estes valores são a população
    # inteira do frame, não uma amostra dela. Com um valor só não há dispersão
    # a relatar, e ``None`` diz isso melhor que 0.0 — que se confundiria com
    # "medi a dispersão e ela é nula", o caso realmente interessante aqui.
    stddev = statistics.pstdev(present) if len(present) > 1 else None
    return ConfidenceStats(
        count=len(present),
        missing=missing,
        minimum=present[0],
        maximum=present[-1],
        mean=sum(present) / len(present),
        median=median,
        distinct_values=distinct,
        stddev=stddev,
        # Uma distribuição degenerada é a que não tem variância nenhuma tendo
        # mais de um valor a comparar. É o que o run de ``corridor-02-002``
        # produziu: 0,9 exato em 60 de 60 regiões. O número é preservado como o
        # produtor o informou; o que este campo faz é tornar a ausência de
        # informação nele legível sem inspecionar a lista inteira.
        degenerate=len(present) > 1 and distinct <= 1,
    )


# Ordena um histograma por contagem decrescente e, em empate, pelo nome. O
# desempate lexicográfico existe para o artifact ser reprodutível: sem ele, dois
# runs idênticos poderiam listar labels empatados em ordens diferentes.
def _histogram(counter: Counter[str]) -> tuple[tuple[str, int], ...]:
    """Retorna o histograma ordenado por contagem decrescente, desempatando por nome."""
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


#: Normalização usada para comparar textos de claim aqui. É a mesma função que
#: o parser usa para deduplicar hipóteses (``domain/semantics.py``): se as duas
#: divergissem, o diagnóstico não mediria o que a deduplicação corrigiu.
_normalized = normalize_claim_value


# Conta quantas regiões repetem, como label próprio, alguma claim da cena. É
# medição e nunca filtro: nenhuma região é descartada por causa deste número.
# Ele existe para tornar comparável entre modos o vazamento de contexto de cena
# descrito na limitação 1 de docs/known-limitations.md.
def _scene_echo_count(observation: VisualObservation) -> int:
    """Conta regiões cujo label primário coincide com uma claim de cena."""
    scene_values = {
        _normalized(claim.value)
        for claim in observation.scene_context.claims
        if claim.value
    }
    if not scene_values:
        return 0
    echoes = 0
    for region in observation.regions:
        claim = primary_label_claim(region)
        if claim is not None and _normalized(claim.value) in scene_values:
            echoes += 1
    return echoes


# Calcula o colapso de modo sobre o histograma de labels primários. O
# denominador é o total de regiões, e não o de regiões rotuladas: a pergunta
# que interessa é "que fração do frame diz a mesma coisa", e uma região sem
# label não pode ser dominante mas continua sendo parte do frame.
def _mode_collapse(
    labels: Counter[str], categories: Counter[str], region_count: int
) -> ModeCollapse:
    """Retorna o label e a categoria dominantes, e a fração do frame que ocupam."""
    dominant_label, dominant_fraction = _dominant(labels, region_count)
    dominant_category, dominant_category_fraction = _dominant(categories, region_count)
    return ModeCollapse(
        dominant_label=dominant_label,
        dominant_fraction=dominant_fraction,
        distinct_labels=len(labels),
        dominant_category=dominant_category,
        dominant_category_fraction=dominant_category_fraction,
        distinct_categories=len(categories),
    )


# Extrai o item mais frequente de um histograma e a fração do frame que ele
# ocupa. Existe para que label e categoria usem exatamente a mesma regra de
# dominância e o mesmo desempate reprodutível.
def _dominant(counter: Counter[str], region_count: int) -> tuple[str | None, float]:
    """Retorna o item dominante e sua fração sobre o total de regiões."""
    if not counter or region_count == 0:
        return None, 0.0
    dominant, count = _histogram(counter)[0]
    return dominant, count / region_count


# Extrai a confiança semântica do label primário de uma região. Isolada em uma
# função para deixar explícito que a ausência vira None, e nunca 0.0 nem a
# confiança geométrica — a confusão exata que este módulo existe para desfazer.
def _semantic_confidence(region: ObservedRegion) -> float | None:
    """Retorna a confiança do label primário da região, ou ``None`` se não houver."""
    claim = primary_label_claim(region)
    if claim is None or claim.confidence is None:
        return None
    return claim.confidence.value


# Conta as regiões que persistem hipóteses de identidade repetidas. Existe
# porque o run de corridor-02-002 gravou "carpet" três vezes na mesma região:
# uma hipótese repetida não é evidência concorrente, e sem este número a
# correção do parser não seria verificável no artifact.
def _duplicate_hypothesis_regions(regions: tuple[ObservedRegion, ...]) -> int:
    """Conta regiões com dois ou mais claims de label de texto equivalente."""
    duplicates = 0
    for region in regions:
        primary = primary_label_claim(region)
        values = [
            _normalized(claim.value)
            for claim in ((primary,) if primary is not None else ()) + alternative_label_claims(region)
        ]
        if len(values) != len(set(values)):
            duplicates += 1
    return duplicates


# Mede quanto de uma máscara de região cai dentro de uma área. Isolada porque
# a mesma razão é usada para o ego e para a área válida, com limiares
# diferentes, e reimplementá-la duas vezes deixaria os dois números
# incomparáveis.
def _overlap_ratio(region: ObservedRegion, area: Mask) -> float:
    """Retorna a fração da máscara da região que cai dentro de ``area``."""
    return _mask_overlap(region.mask, area)


# Mede a fração de uma máscara qualquer que cai dentro de uma área. Existe para
# que a verificação da pós-condição sobre as proposals e a medição sobre as
# regiões usem exatamente a mesma razão.
def _mask_overlap(mask: Mask, area: Mask) -> float:
    """Retorna a fração dos pixels de ``mask`` que caem dentro de ``area``."""
    total = mask.area()
    if total == 0:
        return 0.0
    return float((mask.data & area.data).sum()) / total


# Conta os descartes de proposal por motivo. Existe para que o artifact
# responda "o que foi removido e por quê" sem que o chamador reimplemente a
# agregação — e para que um motivo novo apareça sozinho no histograma.
def _rejection_counts(rejections: tuple[RejectedProposal, ...]) -> Counter[str]:
    """Retorna o histograma de motivos de descarte."""
    return Counter(rejection.reason.value for rejection in rejections)


# Produz o diagnóstico completo de uma observação. Chamada pelo harness de
# validação depois de cada frame, com a contagem de proposals e os descartes
# vindos do PipelineResult — sem eles não há como distinguir over-segmentation
# do SAM de falha do merge geométrico, nem afirmar que a exclusão foi aplicada.
def diagnose_observation(
    observation: VisualObservation,
    *,
    discovered_proposals: int,
    kept_proposals: tuple[RegionProposal, ...] = (),
    proposal_rejections: tuple[RejectedProposal, ...] = (),
    area_masks: ImageAreaMasks | None = None,
    ego_overlap_threshold: float = _DEFAULT_EGO_OVERLAP_THRESHOLD,
    valid_area_threshold: float = _DEFAULT_VALID_AREA_THRESHOLD,
) -> ObservationDiagnostics:
    """Resume uma observação do estágio de proposals até a distribuição de labels.

    Argumentos:
        observation: a observação canônica produzida pelo pipeline.
        discovered_proposals: quantas proposals discovery produziu antes da
            filtragem. Obrigatório, e não derivado da observação, porque um
            default esconderia a origem do número.
        kept_proposals: as proposals que sobreviveram à filtragem, usadas para
            verificar a pós-condição de que nenhuma delas caiu fora da área
            válida ou sobre o rig.
        proposal_rejections: os descartes registrados pela filtragem, usados
            para contar exclusão de ego e de área válida por motivo.
        area_masks: as áreas declaradas do frame, para medir quanto delas ainda
            aparece nas regiões finais. ``None`` significa nenhuma declarada, e
            o diagnóstico registra isso em vez de fingir que houve filtragem.
        ego_overlap_threshold: acima desta fração, uma região final é contada
            como sobreposta ao rig.
        valid_area_threshold: abaixo desta fração dentro da área válida, uma
            região final é contada como fora dela.
    Retorna:
        o diagnóstico do frame, pronto para ser serializado por quem chama.
    """
    regions = observation.regions
    labels: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    region_kinds: Counter[str] = Counter()
    for region in regions:
        claim = primary_label_claim(region)
        # Cada região vota uma vez, com seu label primário. Contar também as
        # alternativas deixaria uma região com quatro hipóteses distorcer o
        # colapso de modo de todo o frame.
        if claim is not None:
            labels[claim.value] += 1
            if claim.category:
                categories[claim.category] += 1
        # Uma região sem interpretação, ou cujo produtor não declarou o kind,
        # conta como ``unknown``. Omiti-la faria a soma do histograma divergir
        # de region_count sem que nada explicasse a diferença.
        kind = RegionKind.UNKNOWN if claim is None or claim.region_kind is None else claim.region_kind
        region_kinds[kind.value] += 1

    rejections = _rejection_counts(proposal_rejections)
    masks = area_masks or ImageAreaMasks()
    ego_mask, valid_mask = masks.ego_vehicle, masks.valid_area
    merged = sum(len(region.contributing_proposal_ids) for region in regions)
    return ObservationDiagnostics(
        region_count=len(regions),
        proposal_count=discovered_proposals,
        # merged_proposal_count conta as proposals efetivamente atribuídas a
        # alguma região. A diferença para proposal_count é exatamente o que a
        # filtragem descartou, e rejected_proposals diz por quê.
        merged_proposal_count=merged,
        regions_with_multiple_proposals=sum(
            1 for region in regions if len(region.contributing_proposal_ids) > 1
        ),
        label_counts=_histogram(labels),
        category_counts=_histogram(categories),
        region_kinds=_histogram(region_kinds),
        semantic_confidence=_summarize(tuple(_semantic_confidence(region) for region in regions)),
        geometric_confidence=_summarize(tuple(region.geometric_confidence for region in regions)),
        duplicate_label_hypotheses=_duplicate_hypothesis_regions(regions),
        mode_collapse=_mode_collapse(labels, categories, len(regions)),
        scene_echo_label_count=_scene_echo_count(observation),
        ego=EgoExclusionStats(
            proposals_rejected=rejections[ProposalRejectionReason.EGO_VEHICLE_OVERLAP.value],
            proposals_overlapping_ego=0
            if ego_mask is None
            else sum(
                1
                for proposal in kept_proposals
                if _mask_overlap(proposal.mask, ego_mask) > ego_overlap_threshold
            ),
            regions_overlapping_ego=0
            if ego_mask is None
            else sum(
                1 for region in regions if _overlap_ratio(region, ego_mask) > ego_overlap_threshold
            ),
            applied=ego_mask is not None,
        ),
        fisheye=ValidAreaStats(
            proposals_rejected=rejections[ProposalRejectionReason.OUTSIDE_VALID_AREA.value],
            proposals_outside_valid_area=0
            if valid_mask is None
            else sum(
                1
                for proposal in kept_proposals
                if _mask_overlap(proposal.mask, valid_mask) < valid_area_threshold
            ),
            regions_outside_valid_area=0
            if valid_mask is None
            else sum(
                1 for region in regions if _overlap_ratio(region, valid_mask) < valid_area_threshold
            ),
            applied=valid_mask is not None,
        ),
        rejected_proposals=_histogram(rejections),
        contextual=_contextual_diagnostics(observation),
    )


# Mede o que os estágios de contexto produziram neste frame. Chamada por
# diagnose_observation; separada porque responde a uma pergunta diferente das
# demais seções — não "o que a geometria encontrou", mas "quanto do que foi
# afirmado tem suporte independente".
def _contextual_diagnostics(observation: VisualObservation) -> ContextualDiagnostics:
    """Resume sinais, reconciliação, grupos e relações semânticas do frame."""
    statuses: Counter[str] = Counter()
    unsupported = ambiguous = contradictions = reconciled = abstained = unscored = 0
    raw_labels: set[str] = set()
    concepts: set[str] = set()

    for region in observation.regions:
        primary = primary_label_claim(region)
        if primary is not None:
            raw_labels.add(_normalized(primary.value))
            if primary.confidence is None:
                unscored += 1
            if region_kind_verdict(primary) is StructuralVerdict.CONTRADICTS:
                contradictions += 1
            measured = [signal for signal in primary.signals if signal.is_measured]
            for signal in primary.signals:
                statuses[signal.status.value] += 1
            if measured:
                if all(item.status is SupportSignalStatus.CONTRADICTS for item in measured):
                    unsupported += 1
                elif all(item.status is SupportSignalStatus.INDISTINGUISHABLE for item in measured):
                    ambiguous += 1
        derived = _reconciled_claim(region)
        if derived is not None:
            reconciled += 1
            concepts.add(_normalized(derived.value))
        elif primary is not None:
            concepts.add(_normalized(primary.value))
        for claim in region.claims:
            if claim.support is not None and claim.support.state is SupportState.ABSTAINED:
                abstained += 1

    scene_values = {
        _normalized(claim.value) for claim in observation.scene_context.claims if claim.value
    }
    echo_any = 0
    competing = 0
    for region in observation.regions:
        asserted = [
            claim
            for claim in region.claims
            if claim.kind is ClaimKind.LABEL and claim.role in ASSERTED_HYPOTHESIS_ROLES
        ]
        if len(asserted) > 1:
            competing += 1
        if scene_values and any(_normalized(claim.value) in scene_values for claim in asserted):
            echo_any += 1

    inferred: Counter[str] = Counter()
    geometric = 0
    for relation in observation.relations:
        if relation.source is RelationSource.MODEL_INFERRED:
            inferred[relation.predicate] += 1
        else:
            geometric += 1

    return ContextualDiagnostics(
        signal_statuses=_histogram(statuses),
        regions_with_unsupported_primary=unsupported,
        regions_with_ambiguous_identity=ambiguous,
        region_kind_contradictions=contradictions,
        reconciled_regions=reconciled,
        distinct_raw_labels=len(raw_labels),
        distinct_canonical_concepts=len(concepts),
        entity_groups=len(observation.entity_hypotheses),
        regions_in_entity_groups=len(
            {member for entity in observation.entity_hypotheses for member in entity.member_region_ids}
        ),
        supported_entity_groups=sum(
            1 for entity in observation.entity_hypotheses if entity.status.value == "supported"
        ),
        semantic_relations=_histogram(inferred),
        geometric_relations=geometric,
        abstained_claims=abstained,
        unscored_claims=unscored,
        scene_echo_any_assertion=echo_any,
        competing_assertions=competing,
    )


# Localiza a claim de identidade reconciliada de uma região. Vive aqui, e não
# importada da reconciliação, porque este módulo é puro e não deve depender de
# um estágio de application só para ler um papel de claim.
def _reconciled_claim(region: ObservedRegion) -> SemanticClaim | None:
    """Retorna a claim reconciliada da região, ou ``None`` se não houver."""
    for claim in region.claims:
        if claim.role is HypothesisRole.RECONCILED:
            return claim
    return None
