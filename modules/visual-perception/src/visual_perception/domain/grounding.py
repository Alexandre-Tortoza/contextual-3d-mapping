"""Evidência espacial condicionada a um conceito, separada de discovery."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Any

from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.identifiers import validate_identifier
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import RegionKind


# Distingue ausência de grounding, falha e geometria validada para que nenhuma
# claim autorize, por omissão, a máscara que serviu apenas para discovery.
class GroundingStatus(StrEnum):
    """Desfecho auditável do grounding de uma hipótese de região."""

    REFINED = "refined"
    FAILED = "grounding_failed"
    UNAVAILABLE = "grounding_unavailable"
    FRAGMENTED = "mask_fragmented"
    TOO_SMALL = "mask_too_small_after_refinement"
    INCONSISTENT = "semantic_mask_inconsistent"


# Entrega ao backend somente o conceito e o suporte de discovery que delimitam
# a busca; não permite que a implementação modifique claims ou regiões.
@dataclass(frozen=True)
class GroundingRequest:
    """Hipótese a localizar na imagem original, em pixels top-left xyxy."""

    region_id: str
    concept: str
    region_kind: RegionKind
    discovery_mask: Mask


# Preserva a saída geométrica bruta do backend antes das políticas de validade
# e conectividade, tornando clipping e rejeições reproduzíveis sem modelos.
@dataclass(frozen=True)
class GroundingPrediction:
    """Máscara do segmentador e box do detector para uma hipótese específica."""

    region_id: str
    concept: str
    model_mask: Mask | None = None
    prompt_box: BoundingBox | None = None
    detection_confidence: float | None = None
    geometric_confidence: float | None = None
    provenance: tuple[ModelProvenance, ...] = ()
    reason: str | None = None
    status: GroundingStatus = GroundingStatus.FAILED
    model_calls: int = 0
    latency_s: float = 0.0
    prompt_boxes: tuple[BoundingBox, ...] = ()

    # Rejeita scores e duração impossíveis na fronteira de um backend externo.
    def __post_init__(self) -> None:
        """Valida medidas e a presença de geometria em uma predição bem-sucedida."""
        if not self.region_id or not self.concept.strip():
            raise ValueError("Grounding requires a region identity and a concept.")
        if not isinstance(self.status, GroundingStatus):
            raise TypeError("Grounding status must use GroundingStatus.")
        for score in (self.detection_confidence, self.geometric_confidence):
            if score is not None and (not isfinite(score) or not 0 <= score <= 1):
                raise ValueError("Grounding scores must be finite values in [0, 1].")
        if not isfinite(self.latency_s) or self.latency_s < 0 or self.model_calls < 0:
            raise ValueError("Grounding cost must be finite and non-negative.")
        if self.status is GroundingStatus.REFINED and (
            self.model_mask is None or self.prompt_box is None or not self.provenance
        ):
            raise ValueError("Refined predictions require mask, prompt box and provenance.")
        boxes = self.prompt_boxes + (() if self.prompt_box is None else (self.prompt_box,))
        if any(not all(isfinite(value) for value in (box.x_min, box.y_min, box.x_max, box.y_max)) for box in boxes):
            raise ValueError("Grounding prompt boxes must have finite coordinates.")


# Anexa a geometria aceita à região sem substituir sua descoberta ou o label.
# O suporte principal é escolhido antes de ownership e nunca muda para salvar
# fragmentos que sobraram depois de uma região prioritária tomar seus pixels.
@dataclass(frozen=True)
class SemanticGrounding:
    """Grounding de um conceito, com máscara bruta, aceita e transformações."""

    prediction: GroundingPrediction
    status: GroundingStatus
    semantic_mask: Mask | None = None
    support_pixel: tuple[int, int] | None = None
    #: Somente leitura no primeiro nível: o grounding é congelado, e um dict
    #: mutável deixava qualquer consumidor reescrever o diagnóstico publicado.
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    # Uma falha não pode transportar uma máscara aparentemente utilizável.
    def __post_init__(self) -> None:
        """Exige que máscara semântica e status concordem, e congela o diagnóstico."""
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if (self.status is GroundingStatus.REFINED) != (self.semantic_mask is not None):
            raise ValueError("Only refined grounding may carry a semantic mask.")
        if self.semantic_mask is not None and self.semantic_mask.is_empty:
            raise ValueError("A semantic mask must not be empty.")
        if self.semantic_mask is not None:
            raw = self.prediction.model_mask
            if raw is None or raw.data.shape != self.semantic_mask.data.shape:
                raise ValueError("Semantic and model mask dimensions must agree.")
            if (self.semantic_mask.data & ~raw.data).any():
                raise ValueError("Semantic mask must be supported by the model mask.")
        if self.support_pixel is not None:
            raw = self.prediction.model_mask
            if (len(self.support_pixel) != 2 or any(type(value) is not int for value in self.support_pixel)
                    or raw is None or not 0 <= self.support_pixel[0] < raw.image_width
                    or not 0 <= self.support_pixel[1] < raw.image_height):
                raise ValueError("Grounding support pixel must lie inside the image.")


# Publica a geometria final após ownership, mantendo o diagnóstico de regiões
# sem footprint; é consumido pela composição que cria VisualRegionEvidence.
@dataclass(frozen=True)
class SpatialRegionFootprint:
    """Suporte espacial de região; ausência de máscara preserva a abstenção."""

    region_id: str
    concept: str
    mask: Mask | None
    strong: bool
    #: Somente leitura no primeiro nível, pela mesma razão de ``SemanticGrounding``.
    diagnostics: Mapping[str, Any]

    # Valida a identidade e a coerência entre força e máscara, e congela o
    # diagnóstico publicado junto do footprint.
    def __post_init__(self) -> None:
        """Rejeita footprint sem identidade ou forte sem máscara, e congela o diagnóstico."""
        validate_identifier(self.region_id, field="region_id")
        if not self.concept.strip():
            raise ValueError("SpatialRegionFootprint requires a non-empty concept.")
        if self.strong and self.mask is None:
            raise ValueError("A strong SpatialRegionFootprint must carry a mask.")
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
