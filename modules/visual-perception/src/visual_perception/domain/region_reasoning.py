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

from visual_perception.domain.geometry import BoundingBox, CoordinateTransform
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.region_evidence import CONTEXTUAL_SLOTS, FOREGROUND_SLOTS, EvidenceSlot
from visual_perception.domain.semantics import ClaimKind, SemanticClaim
from visual_perception.domain.visual_observation import SceneContext

#: As kinds de claim de cena que podem informar a interpretação de uma
#: região. É o subconjunto canônico exigido pela #202: descreve a cena, não
#: uma região dela. Kinds de região (``LABEL``, ``CONDITION``, ``MATERIAL``)
#: ficam de fora por construção, para que nenhuma propriedade global entre no
#: raciocínio de região disfarçada de propriedade local.
REGION_SCENE_CLAIM_KINDS = frozenset(
    {
        ClaimKind.SCENE_TYPE,
        ClaimKind.SCENE_DESCRIPTION,
        ClaimKind.ATTRIBUTE,
        ClaimKind.HAZARD,
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
    masked: bool

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


# Seleciona as claims de cena que podem informar a interpretação de uma
# região, preservando cada claim inteira. Existe porque a #202 proíbe achatar
# a cena em uma string antes do raciocínio de região: confiança, support,
# evidência e proveniência de cada claim continuam individualmente legíveis
# e auditáveis. Chamada por application/region_semantics.py ao montar cada
# RegionReasoningRequest.
def select_region_scene_claims(scene_context: SceneContext | None) -> tuple[SemanticClaim, ...]:
    """Retorna o subconjunto canônico de claims de cena visível ao raciocínio de região.

    Claims contraditórias são preservadas lado a lado: a seleção não escolhe
    entre hipóteses de cena, apenas restringe as kinds que atravessam a
    fronteira. Contexto ausente devolve uma tupla vazia, que é uma entrada
    válida — a região é interpretada apenas pela sua própria evidência.

    Argumentos:
        scene_context: o contexto de cena da observação, ou ``None``.
    Retorna:
        as claims de cena elegíveis, na ordem em que a cena as produziu.
    """
    if scene_context is None:
        return ()
    return tuple(claim for claim in scene_context.claims if claim.kind in REGION_SCENE_CLAIM_KINDS)
