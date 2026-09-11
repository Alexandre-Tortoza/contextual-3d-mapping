"""Entrada canônica do raciocínio semântico de região.

Issues: #202 (contexto de cena estruturado) e #203 (evidência mask-aware
multi-contexto).

Antes desta fronteira, a interpretação de região recebia dois argumentos
soltos: um recorte retangular pelo bounding box e a *primeira* string de
``scene_description``. Os dois eram redutores. O recorte pela caixa mostra
tudo que está dentro do retângulo — em regiões finas, diagonais ou
recortadas, a maior parte desses pixels é fundo, e o label acabava
descrevendo o fundo. E achatar a cena em uma string descartava scene type,
atributos, hazards, provenance e support, exatamente a evidência que a #202
exige preservar.

:class:`RegionReasoningRequest` substitui esse par por um objeto único que
carrega, com identidade de região estável:

- as :class:`RegionView` em pixels, distinguíveis por slot, de forma que
  foreground e contexto cheguem separados e não misturados em um recorte só;
- as claims de cena estruturadas, cada uma com a sua própria confiança,
  support e proveniência.

A geometria é imutável aqui: cada view carrega a caixa e o transform que a
produziram, e nada nesta fronteira pode alterar mask, box ou identidade da
região.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import sqrt

from visual_perception.domain.geometry import BoundingBox, CoordinateTransform
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import (
    CONTEXTUAL_SLOTS,
    FOREGROUND_SLOTS,
    EvidenceSlot,
    SubjectEmphasis,
)
from visual_perception.domain.semantics import ClaimKind, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext


#: As kinds de claim de cena que podem informar a interpretação de uma
#: região. É o subconjunto canônico exigido pela #202: descreve a cena, não
#: uma região dela. Kinds de região (``LABEL``, ``CONDITION``, ``MATERIAL``)
#: ficam de fora por construção, para que nenhuma propriedade global entre no
#: raciocínio de região disfarçada de propriedade local.
# Enumera se o raciocínio de uma região enxerga o contexto de cena. Existe
# porque proibir textualmente a repetição da cena foi tentado e medido, e
# falhou: o prompt já afirma que as claims descrevem a cena e "must not be
# repeated as the label", e ainda assim regiões saem rotuladas com a cena
# inteira (docs/known-limitations.md, limitação 1). A única garantia estrutural
# é o contexto não entrar no request. Selecionado por
# MultimodalReasoningConfig.scene_context_mode e aplicado em
# select_region_scene_claims.
class SceneContextMode(StrEnum):
    """Se as claims de cena atravessam a fronteira do raciocínio de região."""

    #: A região é interpretada apenas pela sua própria evidência local.
    LOCAL_FIRST = "local_first"
    #: As claims de cena acompanham a região, para desambiguação.
    CONTEXT_ASSISTED = "context_assisted"


# Enumera se o raciocínio de uma região enxerga o que o frame anterior afirmou
# sobre a mesma área da imagem. Existe pelo mesmo motivo que SceneContextMode:
# a única garantia estrutural contra o modelo copiar uma sugestão é a sugestão
# não entrar no request. Selecionado por
# MultimodalReasoningConfig.temporal_prior_mode e aplicado em
# select_region_prior.
class TemporalPriorMode(StrEnum):
    """Se a observação anterior da mesma área atravessa a fronteira do raciocínio."""

    #: Cada frame é interpretado sozinho. É o comportamento histórico.
    DISABLED = "disabled"
    #: A região recebe o conceito afirmado antes para a área que ela cobre,
    #: casada por sobreposição de caixa.
    BOX_OVERLAP = "box_overlap"


# Descreve uma região que o frame anterior afirmou, reduzida ao mínimo que o
# casamento e o prompt precisam. Existe para que o prior seja um value object
# fechado em vez de um ``ObservedRegion`` inteiro: reter a região anterior
# traria máscara em resolução plena entre frames, que é exatamente o que
# docs/api-contracts.md desaconselha ao proibir reter um PipelineResult.
@dataclass(frozen=True)
class PriorRegion:
    """Uma região afirmada pelo frame anterior, candidata a informar o atual."""

    region_id: str
    box: BoundingBox
    concept: str
    category: str | None = None
    #: Embedding denso da região anterior, usado **apenas** para desempatar
    #: candidatos que já se sobrepõem. Vazio quando o slot denso não rodou.
    embedding: tuple[float, ...] = field(default_factory=tuple)

    # Rejeita um prior sem identidade ou sem conceito: os dois são obrigatórios
    # para que a sugestão seja atribuível a uma observação anterior concreta.
    def __post_init__(self) -> None:
        """Valida identidade e conceito da região anterior."""
        validate_identifier(self.region_id, field="region_id")
        if not self.concept.strip():
            raise ValueError(f"PriorRegion({self.region_id!r}) requires a non-empty concept.")


# Reúne o que o frame anterior afirmou, sem nenhum metadado de sequência.
# Existe para que ``visual-perception`` continue sem conhecer tempo, pose ou
# gravação: quem sabe que um frame vem antes de outro é a composição, e o
# módulo recebe apenas "isto foi afirmado antes", nunca "isto foi afirmado em t".
@dataclass(frozen=True)
class ScenePrior:
    """As regiões afirmadas pela observação anterior da mesma cena."""

    regions: tuple[PriorRegion, ...] = field(default_factory=tuple)

    # Rejeita identidades repetidas, que tornariam o casamento não determinístico.
    def __post_init__(self) -> None:
        """Valida que cada região anterior aparece uma única vez."""
        identifiers = [region.region_id for region in self.regions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("ScenePrior must not contain repeated region ids.")


# Carrega a sugestão que uma região recebeu, junto da evidência que justificou
# o casamento. Existe porque "esta região recebeu um prior" precisa ser
# auditável depois do fato: sem overlap e similarity registrados, não haveria
# como distinguir um prior bem ancorado de um casamento de sorte.
@dataclass(frozen=True)
class PriorHypothesis:
    """O que o frame anterior afirmou sobre a área que esta região cobre."""

    concept: str
    source_region_id: str
    overlap: float
    category: str | None = None
    #: Cosseno entre os embeddings densos, quando ambos existem. ``None``
    #: significa que o desempate não teve como ser feito, e não similaridade zero.
    similarity: float | None = None


#: As claims de cena que podem acompanhar uma região no prompt: **apenas** as
#: ambientais. ``SCENE_DESCRIPTION``, ``ATTRIBUTE`` e ``HAZARD`` saíram na #202
#: porque descrevem um inventário de objetos, e um objeto inferido globalmente
#: não pode induzir a identidade de uma região local — em ``corridor-02-002``
#: era por ali que "there is a suitcase in the foreground", que era o próprio
#: rig, alcançava o prompt de cada uma das 60 regiões.
REGION_SCENE_CLAIM_KINDS = frozenset(
    {
        ClaimKind.SCENE_TYPE,
        ClaimKind.ENVIRONMENT,
        ClaimKind.LAYOUT,
        ClaimKind.LIGHTING,
        ClaimKind.VISIBILITY,
        ClaimKind.NAVIGABILITY,
    }
)


# Representa uma visão em pixels de uma região, amarrada à geometria que a
# produziu. Existe porque a #203 exige que foreground e contexto cheguem ao
# reasoner como evidências distinguíveis: o slot diz o que a view é, e
# crop_box/transform permitem voltar de qualquer pixel da view para a
# coordenada original da imagem.
@dataclass(frozen=True)
class RegionView:
    """Uma visão em pixels de uma região, com o slot e a geometria que a originaram."""

    slot: EvidenceSlot
    payload: ImagePayload
    crop_box: BoundingBox
    transform: CoordinateTransform
    #: Como esta view torna o sujeito identificável. Substituiu um booleano
    #: ``masked`` na #202: ele não conseguia expressar o crop contextual, que
    #: demarca o sujeito pelo contorno sem suprimir o entorno.
    emphasis: SubjectEmphasis = SubjectEmphasis.NONE

    # Impõe que a view descreva exatamente a caixa que diz descrever. Sem
    # isso, um recorte fora de sincronia com a sua caixa faria o reasoner
    # olhar para uma região e o consumidor atribuir o resultado a outra.
    def __post_init__(self) -> None:
        """Valida que os pixels da view têm a resolução da sua ``crop_box``."""
        expected_width = int(self.crop_box.x_max) - int(self.crop_box.x_min)
        expected_height = int(self.crop_box.y_max) - int(self.crop_box.y_min)
        if self.payload.width != expected_width or self.payload.height != expected_height:
            raise ValueError(
                f"RegionView {self.slot.value!r} has a {self.payload.width}x{self.payload.height} "
                f"payload but a {expected_width}x{expected_height} crop_box."
            )

    # Responde se o tratamento suprimiu tudo fora da máscara. Existe como
    # propriedade derivada para que os consumidores anteriores à #202
    # continuem lendo a mesma informação sem conhecer SubjectEmphasis.
    @property
    def masked(self) -> bool:
        """Indica que a view suprimiu os pixels fora da máscara da região."""
        return self.emphasis.is_masked

    # Responde se a view mostra apenas o objeto, sem entorno. Reusa a
    # classificação de slots da #194 para que consumidores não reimplementem
    # a separação foreground/contexto.
    @property
    def is_foreground(self) -> bool:
        """Indica se a view cobre apenas o objeto da região."""
        return self.slot in FOREGROUND_SLOTS


# Agrupa tudo que o reasoner multimodal precisa para interpretar uma região:
# identidade estável, geometria canônica, as views distinguíveis e o contexto
# de cena estruturado. Existe para substituir o par
# ``(crop, scene_summary)`` do port anterior por uma entrada única e
# auditável; construída por application/region_semantics.py e consumida pelos
# adapters de MultimodalReasoner.
@dataclass(frozen=True)
class RegionReasoningRequest:
    """A entrada canônica de uma interpretação semântica de região."""

    region_id: str
    region_box: BoundingBox
    image_width: int
    image_height: int
    views: tuple[RegionView, ...]
    scene_claims: tuple[SemanticClaim, ...] = field(default_factory=tuple)
    #: O que a observação anterior afirmou sobre a área que esta região cobre.
    #: ``None`` é o caso normal e o default: sem prior, o request é idêntico ao
    #: histórico, o que mantém os runs anteriores comparáveis.
    prior: PriorHypothesis | None = None

    # Impõe as invariantes que tornam o request interpretável: identidade
    # válida, pelo menos uma view de foreground (sem ela não há evidência
    # local, e interpretar seria descrever o entorno), e um slot por view,
    # para que "a view de contexto" seja sempre não-ambígua.
    def __post_init__(self) -> None:
        """Valida identidade, presença de foreground e unicidade de slot entre as views."""
        validate_identifier(self.region_id, field="region_id")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("RegionReasoningRequest requires a positive image resolution.")
        if not self.views:
            raise ValueError(f"Region {self.region_id!r} has no evidence view to reason about.")
        slots = [view.slot for view in self.views]
        if len(set(slots)) != len(slots):
            raise ValueError(f"Region {self.region_id!r} has more than one view per evidence slot.")
        if not any(view.is_foreground for view in self.views):
            raise ValueError(
                f"Region {self.region_id!r} has only contextual views: interpreting it would "
                "describe the surroundings instead of the region."
            )

    # Expõe apenas as views que mostram o objeto. Usada pelos adapters para
    # ordenar o prompt do mais local para o mais global, sem conhecer o
    # conjunto concreto de slots.
    @property
    def foreground_views(self) -> tuple[RegionView, ...]:
        """As views que cobrem apenas o objeto da região."""
        return tuple(view for view in self.views if view.slot in FOREGROUND_SLOTS)

    # Expõe apenas as views que incluem entorno ou cena, complementares às de
    # foreground e sempre separadas delas.
    @property
    def contextual_views(self) -> tuple[RegionView, ...]:
        """As views que incluem entorno ou cena."""
        return tuple(view for view in self.views if view.slot in CONTEXTUAL_SLOTS)


# Agrupa tudo que o reasoner precisa para julgar a relação entre **duas**
# regiões canônicas: as identidades já reconciliadas, a natureza de cada uma, o
# que a geometria já mediu sobre o par, e uma view em pixels que mostra as duas
# demarcadas dentro do mesmo recorte. Existe separado de
# :class:`RegionReasoningRequest` porque a pergunta é outra — não "o que é esta
# região", mas "como estas duas se relacionam" — e um request que servisse às
# duas teria campos opcionais que só um dos caminhos preenche.
#
# ``geometric_summary`` carrega o que o caminho geométrico **já mediu** (quanto
# uma contém a outra, quanto se sobrepõem). Ele existe para que o modelo não
# precise reestimar no olho uma grandeza que o módulo calculou exatamente, e
# nunca para substituir a evidência visual: um par sem view não é interpretável.
@dataclass(frozen=True)
class RegionRelationRequest:
    """A entrada canônica de uma inferência de relação entre duas regiões."""

    subject_region_id: str
    object_region_id: str
    subject_concept: str
    object_concept: str
    views: tuple[RegionView, ...]
    geometric_summary: str
    subject_kind: str = "unknown"
    object_kind: str = "unknown"

    # Impõe as invariantes que tornam a resposta atribuível: dois sujeitos
    # distintos e válidos, conceitos não vazios, e ao menos uma view — sem ela
    # o modelo estaria julgando a relação só pelo texto que lhe demos.
    def __post_init__(self) -> None:
        """Valida identidades, conceitos e presença de evidência visual do par."""
        validate_identifier(self.subject_region_id, field="subject_region_id")
        validate_identifier(self.object_region_id, field="object_region_id")
        if self.subject_region_id == self.object_region_id:
            raise ValueError(
                f"RegionRelationRequest({self.subject_region_id!r}) relates a region to itself."
            )
        if not self.subject_concept.strip() or not self.object_concept.strip():
            raise ValueError("RegionRelationRequest requires a non-empty concept for both regions.")
        if not self.views:
            raise ValueError(
                f"RegionRelationRequest({self.subject_region_id!r}, {self.object_region_id!r}) has no "
                "view: judging a relation from text alone would not be visual evidence."
            )
        if not self.geometric_summary.strip():
            raise ValueError("RegionRelationRequest requires a non-empty geometric_summary.")


# Seleciona as claims de cena que podem informar a interpretação de uma
# região, preservando cada claim inteira. Existe porque a #202 proíbe achatar
# a cena em uma string antes do raciocínio de região: confiança, support,
# evidência e proveniência de cada claim continuam individualmente legíveis
# e auditáveis. Chamada por application/region_semantics.py ao montar cada
# RegionReasoningRequest.
def select_region_scene_claims(
    scene_context: SceneContext | None,
    *,
    mode: SceneContextMode = SceneContextMode.CONTEXT_ASSISTED,
) -> tuple[SemanticClaim, ...]:
    """Retorna o subconjunto canônico de claims de cena visível ao raciocínio de região.

    Claims contraditórias são preservadas lado a lado: a seleção não escolhe
    entre hipóteses de cena, apenas restringe as kinds que atravessam a
    fronteira. Contexto ausente devolve uma tupla vazia, que é uma entrada
    válida — a região é interpretada apenas pela sua própria evidência.

    Em ``LOCAL_FIRST`` a tupla é vazia por decisão de modo, e não por ausência
    de cena: a cena continua sendo analisada e permanece na observação, apenas
    não atravessa esta fronteira.

    Argumentos:
        scene_context: o contexto de cena da observação, ou ``None``.
        mode: se o contexto de cena acompanha a região.
    Retorna:
        as claims de cena elegíveis, na ordem em que a cena as produziu.
    """
    if mode is SceneContextMode.LOCAL_FIRST:
        return ()
    if scene_context is None:
        return ()
    return tuple(claim for claim in scene_context.claims if claim.kind in REGION_SCENE_CLAIM_KINDS)


# Calcula o cosseno entre dois embeddings densos. Existe local e privada porque
# é usada só para desempatar candidatos que já se sobrepõem: promover isso a
# utilitário público sugeriria que o módulo tem um comparador de embeddings
# entre frames, que é justamente o que a #203 mediu como insuficiente sozinho
# (acurácia balanceada de 0,660 para separar "mesmo label" de "label diferente").
def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    """Retorna o cosseno entre dois vetores, ou ``None`` se algum for vazio ou nulo."""
    if not left or not right or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(a * a for a in left))
    right_norm = sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


# Casa uma região com o que o frame anterior afirmou sobre a mesma área da
# imagem. Existe como par de select_region_scene_claims: os dois decidem o que
# atravessa a fronteira do raciocínio, um no eixo da cena e outro no eixo do
# tempo, e os dois devolvem vazio por decisão de modo em vez de por ausência.
#
# O casamento é por **sobreposição de caixa**, e não por similaridade de
# embedding, por uma razão medida: a #203 registra que o cosseno DINOv2 separa
# "mesmo label" de "label diferente" com acurácia balanceada de apenas 0,660, e
# o módulo o usa como corroboração e nunca como gate. Em amostragem densa a
# câmera mal se desloca entre frames vizinhos, e a sobreposição de caixa é o
# sinal forte nesse regime; o embedding entra só para desempatar candidatos que
# já se sobrepõem, que é exatamente o papel de corroboração que ele sustenta.
def select_region_prior(
    region_box: BoundingBox,
    prior: ScenePrior | None,
    *,
    mode: TemporalPriorMode = TemporalPriorMode.DISABLED,
    min_overlap: float = 0.3,
    embedding: tuple[float, ...] = (),
) -> PriorHypothesis | None:
    """Retorna o que a observação anterior afirmou sobre a área desta região.

    Argumentos:
        region_box: caixa da região no frame atual.
        prior: o que a observação anterior afirmou, ou ``None``.
        mode: se a observação anterior acompanha a região.
        min_overlap: sobreposição mínima de caixa para o casamento valer.
        embedding: embedding denso da região atual, usado só no desempate.
    Retorna:
        a hipótese anterior casada, ou ``None`` quando nada se sobrepõe o
        bastante — que é uma saída normal, e não um erro.
    Levanta:
        ValueError: se ``min_overlap`` estiver fora de ``[0, 1]``.
    """
    if not 0.0 <= min_overlap <= 1.0:
        raise ValueError("min_overlap must be within [0, 1].")
    if mode is TemporalPriorMode.DISABLED or prior is None:
        return None
    candidates = [
        (region_box.iou(previous.box), previous)
        for previous in prior.regions
        if region_box.iou(previous.box) >= min_overlap
    ]
    if not candidates:
        return None
    # Ordena por similaridade densa e só depois por sobreposição, mas apenas
    # entre candidatos que já passaram no gate geométrico. A ordenação final por
    # region_id existe para que dois candidatos empatados em tudo produzam
    # sempre o mesmo resultado.
    scored = [
        (overlap, _cosine(embedding, previous.embedding), previous) for overlap, previous in candidates
    ]
    overlap, similarity, chosen = max(
        scored,
        key=lambda item: (
            item[1] if item[1] is not None else -2.0,
            item[0],
            item[2].region_id,
        ),
    )
    return PriorHypothesis(
        concept=chosen.concept,
        source_region_id=chosen.region_id,
        overlap=overlap,
        category=chosen.category,
        similarity=similarity,
    )
