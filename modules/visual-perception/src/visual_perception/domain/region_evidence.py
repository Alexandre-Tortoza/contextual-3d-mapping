"""Contract de evidência multi-contexto por região.

Issue: #193.

Uma região não é bem representada por um único embedding. Uma máscara
apertada descreve o objeto e perde o entorno que o desambigua; um crop com
contexto descreve o entorno e dilui o objeto. Em vez de escolher um dos dois
e perder o outro, uma região retém *slots* complementares de evidência:

- ``foreground_dense``: features densas agregadas somente sob a máscara;
- ``tight_crop``: representação alinhada a linguagem do crop justo;
- ``contextual_crop``: o mesmo crop expandido por uma margem de contexto;
- ``scene_conditioned``: evidência condicionada ao contexto de cena.

Cada slot carrega sua própria geometria de imagem, proveniência de
pré-processamento e identidade de espaço de embedding
(:class:`~visual_perception.domain.embeddings.EmbeddingSpace`), para que
slots incompatíveis nunca sejam comparados ou agregados silenciosamente.

Um slot ausente ou que falhou é representado explicitamente (``missing`` /
``failed`` com ``reason``), nunca omitido: a ausência de contexto opcional
não invalida a evidência de foreground que deu certo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from visual_perception.domain.embeddings import EmbeddingModality, EmbeddingSpace
from visual_perception.domain.geometry import BoundingBox, CoordinateTransform
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import SourceArtifactReference


# Enumera os slots estáveis de evidência de uma região. Existe para que o
# conjunto de representações complementares seja um vocabulário fechado e
# versionado, e não uma string livre por produtor.
class EvidenceSlot(StrEnum):
    """Os slots complementares de evidência que uma região pode reter."""

    FOREGROUND_DENSE = "foreground_dense"
    MASKED_SUBJECT = "masked_subject"
    TIGHT_CROP = "tight_crop"
    CONTEXTUAL_CROP = "contextual_crop"
    SCENE_CONDITIONED = "scene_conditioned"


# Distingue evidência presente, deliberadamente ausente e falha. Existe
# porque a #193 exige que contexto opcional faltante seja explícito: um slot
# ``missing`` é uma informação, não um buraco no dado.
class EvidenceState(StrEnum):
    """Se um slot tem evidência, foi deliberadamente pulado, ou falhou."""

    AVAILABLE = "available"
    MISSING = "missing"
    FAILED = "failed"


# Enumera como o sujeito é tornado identificável em uma view. Existe porque a
# evidência que o reasoner recebe tem três tratamentos genuinamente distintos —
# recorte cru, sujeito isolado sobre fundo neutro, e sujeito demarcado dentro do
# contexto — e um booleano ``masked`` não conseguia expressar o terceiro, que é
# justamente o que faltava para o crop contextual deixar de ser ambíguo.
class SubjectEmphasis(StrEnum):
    """Como uma view torna o sujeito da região identificável."""

    NONE = "none"
    ZERO_FILL = "zero_fill"
    NEUTRAL_FILL = "neutral_fill"
    CONTOUR = "contour"

    # Indica se o tratamento suprime o fundo, informação que o RegionView
    # carrega para o reasoner saber se o que ele vê inclui entorno.
    @property
    def is_masked(self) -> bool:
        """Indica que o tratamento suprimiu tudo fora da máscara."""
        return self in {SubjectEmphasis.ZERO_FILL, SubjectEmphasis.NEUTRAL_FILL}

    # Nomeia o tratamento no campo ``preprocessing`` do slot persistido. Existe
    # para que o artifact diga exatamente que preprocessamento produziu aqueles
    # pixels, com um vocabulário estável que o leitor reconhece.
    @property
    def preprocessing_label(self) -> str:
        """Retorna o nome do tratamento usado em ``RegionEvidenceSlot.preprocessing``."""
        return {
            SubjectEmphasis.NONE: "crop",
            SubjectEmphasis.ZERO_FILL: "masked_crop",
            SubjectEmphasis.NEUTRAL_FILL: "neutral_masked_crop",
            SubjectEmphasis.CONTOUR: "contour_crop",
        }[self]


#: Slots cuja evidência descreve apenas o objeto, sem entorno. Usada por
#: consumidores que precisam separar foreground de contexto sem reimplementar
#: a classificação (critério de aceitação da #194).
FOREGROUND_SLOTS = frozenset(
    {EvidenceSlot.FOREGROUND_DENSE, EvidenceSlot.MASKED_SUBJECT, EvidenceSlot.TIGHT_CROP}
)

#: Slots cuja evidência inclui entorno ou cena, complementares aos de foreground.
CONTEXTUAL_SLOTS = frozenset({EvidenceSlot.CONTEXTUAL_CROP, EvidenceSlot.SCENE_CONDITIONED})


# Representa uma unidade de evidência de uma região em um slot específico.
# Existe como o átomo do modelo multi-contexto: cada slot é auditável até a
# sua geometria de imagem, seu pré-processamento e o espaço vetorial em que
# o vetor resultante vive.
@dataclass(frozen=True)
class RegionEvidenceSlot:
    """Uma representação de região em um slot, com geometria e espaço próprios.

    ``artifact_ref`` referencia o vetor por identidade, nunca o embute: o
    contract canônico serializa referências, e os vetores densos viajam
    como artifacts (ver ``docs/artifacts.md``).
    """

    slot: EvidenceSlot
    region_id: str
    state: EvidenceState
    crop_box: BoundingBox | None = None
    transform: CoordinateTransform | None = None
    preprocessing: str | None = None
    artifact_ref: str | None = None
    space: EmbeddingSpace | None = None
    mask_ref: str | None = None
    support_ratio: float | None = None
    #: Fração do bounding box que a máscara ocupa. Existe para tornar auditável
    #: o caso em que a caixa domina semanticamente a região: uma máscara fina e
    #: diagonal preenche pouco do seu box, e uma interpretação feita sobre o box
    #: estaria descrevendo o fundo. ``None`` quando o slot não recorta nada.
    mask_fill_ratio: float | None = None
    reason: str | None = None
    view_id: str | None = None
    source_artifact_refs: tuple[SourceArtifactReference, ...] = field(default_factory=tuple)

    # Impõe as invariantes que separam evidência real de ausência: um slot
    # disponível precisa de geometria, artifact e espaço; um slot ausente ou
    # falho precisa de um motivo e não pode fingir carregar um vetor.
    def __post_init__(self) -> None:
        """Valida a coerência entre estado, geometria, artifact e espaço."""
        validate_identifier(self.region_id, field="region_id")
        if self.view_id is not None:
            validate_identifier(self.view_id, field="view_id")
        if self.support_ratio is not None and not 0.0 <= self.support_ratio <= 1.0:
            raise ValueError(f"RegionEvidenceSlot.support_ratio must be in [0, 1], got {self.support_ratio}.")
        if self.mask_fill_ratio is not None and not 0.0 <= self.mask_fill_ratio <= 1.0:
            raise ValueError(
                f"RegionEvidenceSlot.mask_fill_ratio must be in [0, 1], got {self.mask_fill_ratio}."
            )
        if self.state is EvidenceState.AVAILABLE:
            if self.artifact_ref is None or self.space is None:
                raise ValueError(
                    f"An available {self.slot.value!r} evidence slot requires artifact_ref and space."
                )
            if self.crop_box is None or self.transform is None:
                raise ValueError(
                    f"An available {self.slot.value!r} evidence slot requires crop_box and transform."
                )
            if self.reason is not None:
                raise ValueError("An available evidence slot must not carry a failure reason.")
        else:
            if not self.reason:
                raise ValueError(f"A {self.state.value!r} evidence slot must explain itself with a reason.")
            if self.artifact_ref is not None:
                raise ValueError(f"A {self.state.value!r} evidence slot must not reference an artifact.")

    # Responde se este slot descreve apenas o objeto. Existe para que o
    # consumidor distinga foreground de contexto sem conhecer o conjunto de
    # slots, satisfazendo o critério de distinguibilidade da #194.
    @property
    def is_foreground(self) -> bool:
        """Indica se a evidência deste slot cobre apenas o objeto."""
        return self.slot in FOREGROUND_SLOTS


# Impõe que todos os slots disponíveis de um conjunto vivam no mesmo espaço
# vetorial antes de qualquer comparação ou agregação. Existe porque a #193
# proíbe explicitamente que espaços incompatíveis sejam misturados em
# silêncio; usada por consumidores que agregam evidência entre slots.
def require_comparable_slots(slots: tuple[RegionEvidenceSlot, ...]) -> EmbeddingSpace | None:
    """Retorna o espaço comum dos slots disponíveis, ou ``None`` se não houver nenhum.

    Argumentos:
        slots: os slots de evidência a serem comparados entre si.
    Retorna:
        o :class:`EmbeddingSpace` compartilhado, ou ``None`` sem slots disponíveis.
    Levanta:
        IncompatibleEmbeddingSpaceError: se dois slots declararem espaços diferentes.
    """
    spaces = [slot.space for slot in slots if slot.state is EvidenceState.AVAILABLE and slot.space]
    if not spaces:
        return None
    reference = spaces[0]
    for space in spaces[1:]:
        reference.require_compatible(space)
    return reference


# Seleciona os slots de um estado específico, preservando a ordem. Existe
# para que auditoria e avaliação (#199) enumerem falhas ou ausências por
# região sem replicar o filtro em cada consumidor.
def slots_in_state(
    slots: tuple[RegionEvidenceSlot, ...], state: EvidenceState
) -> tuple[RegionEvidenceSlot, ...]:
    """Filtra os slots que estão em ``state``, preservando a ordem original."""
    return tuple(slot for slot in slots if slot.state is state)


# Converte um slot de evidência em um dict serializável, achatando
# geometria, transform, espaço e referências de artifact. Usada por
# infrastructure/serialization.py ao gravar uma ObservedRegion (#172).
def evidence_to_dict(slot: RegionEvidenceSlot) -> dict[str, Any]:
    """Converte um :class:`RegionEvidenceSlot` em um dict serializável."""
    return {
        "slot": slot.slot.value,
        "region_id": slot.region_id,
        "state": slot.state.value,
        "crop_box": None if slot.crop_box is None else _box_to_dict(slot.crop_box),
        "transform": None if slot.transform is None else _transform_to_dict(slot.transform),
        "preprocessing": slot.preprocessing,
        "artifact_ref": slot.artifact_ref,
        "space": None if slot.space is None else _space_to_dict(slot.space),
        "mask_ref": slot.mask_ref,
        "support_ratio": slot.support_ratio,
        "mask_fill_ratio": slot.mask_fill_ratio,
        "reason": slot.reason,
        "view_id": slot.view_id,
        "source_artifact_refs": [vars(reference) for reference in slot.source_artifact_refs],
    }


# Reconstrói um slot de evidência a partir do dict — lado inverso de
# evidence_to_dict, revalidando todas as invariantes em vez de confiar no
# que estava gravado.
def evidence_from_dict(payload: dict[str, Any]) -> RegionEvidenceSlot:
    """Reconstrói um :class:`RegionEvidenceSlot` validado a partir de um dict."""
    crop_box = payload.get("crop_box")
    transform = payload.get("transform")
    space = payload.get("space")
    return RegionEvidenceSlot(
        slot=EvidenceSlot(payload["slot"]),
        region_id=payload["region_id"],
        state=EvidenceState(payload["state"]),
        crop_box=None if crop_box is None else BoundingBox(**crop_box),
        transform=None if transform is None else CoordinateTransform(**transform),
        preprocessing=payload.get("preprocessing"),
        artifact_ref=payload.get("artifact_ref"),
        space=None if space is None else _space_from_dict(space),
        mask_ref=payload.get("mask_ref"),
        support_ratio=payload.get("support_ratio"),
        mask_fill_ratio=payload.get("mask_fill_ratio"),
        reason=payload.get("reason"),
        view_id=payload.get("view_id"),
        source_artifact_refs=tuple(
            SourceArtifactReference(**reference) for reference in payload.get("source_artifact_refs", ())
        ),
    )


# Achata uma BoundingBox nos quatro floats do contract, sem depender de
# asdict, que traria campos derivados. Helper interno de evidence_to_dict.
def _box_to_dict(box: BoundingBox) -> dict[str, float]:
    """Converte uma :class:`BoundingBox` em seus quatro limites."""
    return {"x_min": box.x_min, "y_min": box.y_min, "x_max": box.x_max, "y_max": box.y_max}


# Achata um CoordinateTransform nos seus quatro coeficientes afins. Helper
# interno de evidence_to_dict.
def _transform_to_dict(transform: CoordinateTransform) -> dict[str, float]:
    """Converte um :class:`CoordinateTransform` em seus coeficientes afins."""
    return {
        "scale_x": transform.scale_x,
        "scale_y": transform.scale_y,
        "offset_x": transform.offset_x,
        "offset_y": transform.offset_y,
    }


# Achata um EmbeddingSpace preservando a modalidade como string. Helper
# interno de evidence_to_dict.
def _space_to_dict(space: EmbeddingSpace) -> dict[str, Any]:
    """Converte um :class:`EmbeddingSpace` em um dict serializável."""
    return {
        "model_id": space.model_id,
        "checkpoint": space.checkpoint,
        "dimension": space.dimension,
        "modality": space.modality.value,
        "normalized": space.normalized,
    }


# Reconstrói um EmbeddingSpace validado a partir do dict. Helper interno de
# evidence_from_dict.
def _space_from_dict(payload: dict[str, Any]) -> EmbeddingSpace:
    """Reconstrói um :class:`EmbeddingSpace` validado a partir de um dict."""
    return EmbeddingSpace(
        model_id=payload["model_id"],
        checkpoint=payload["checkpoint"],
        dimension=payload["dimension"],
        modality=EmbeddingModality(payload["modality"]),
        normalized=payload.get("normalized", True),
    )
