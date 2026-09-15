"""Validação de grounding e ownership sem promover discovery a semântica."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
from scipy import ndimage  # type: ignore[import-untyped]

from visual_perception.config import SemanticGroundingConfig
from visual_perception.domain.errors import BackendUnavailableError, VisualPerceptionError
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.grounding import (
    GroundingPrediction,
    GroundingRequest,
    GroundingStatus,
    SemanticGrounding,
    SpatialRegionFootprint,
)
from visual_perception.domain.image_area import ImageAreaMasks
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.regions import ObservedRegion, primary_label_claim
from visual_perception.domain.semantics import RegionKind
from visual_perception.ports.semantic_grounding import SemanticGrounder


# Centraliza a conectividade 8, que considera contornos diagonais contínuos.
# A área nunca escolhe o componente que receberá uma identidade contável.
def mask_components(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Retorna ids de componentes e sua contagem no frame de imagem."""
    labels, count = ndimage.label(data, structure=np.ones((3, 3), dtype=np.uint8))
    return labels, int(count)


# Registra boxes também para regiões vazias, onde BoundingBox não é definido.
def mask_box(data: np.ndarray) -> dict[str, float] | None:
    """Retorna a box semiaberta ou None quando não há pixels."""
    if not data.any():
        return None
    return asdict(Mask(data, data.shape[1], data.shape[0]).bounding_box())


# Rasteriza a box do detector sem alargar seu alcance para além da imagem.
def box_mask(box: BoundingBox, shape: tuple[int, ...]) -> np.ndarray:
    """Marca os centros de pixels cobertos pela box condicionada ao conceito."""
    y, x = np.indices(shape)
    return np.asarray((x >= box.x_min) & (x < box.x_max) & (y >= box.y_min) & (y < box.y_max), dtype=np.bool_)


# Escolhe suporte próximo ao centro da localização semântica, no segmento SAM.
# Empates entre componentes se abstêm; nunca migra para o maior componente.
def _principal_support(data: np.ndarray, box: BoundingBox) -> tuple[int, int] | None:
    """Retorna o pixel segmentado mais próximo do centro da box do detector."""
    y, x = np.nonzero(data & box_mask(box, data.shape))
    if not len(x):
        return None
    distance = (x - (box.x_min + box.x_max) / 2) ** 2 + (y - (box.y_min + box.y_max) / 2) ** 2
    closest = np.flatnonzero(np.isclose(distance, distance.min(), rtol=0, atol=1e-9))
    components, _ = mask_components(data)
    if len(set(components[y[closest], x[closest]].tolist())) != 1:
        return None
    index = int(closest[0])
    return int(x[index]), int(y[index])


# Descreve cada resto antes da escolha por suporte; área, box e overlaps ficam
# disponíveis para auditar por que o componente escolhido não era o maior.
def _component_diagnostics(
    data: np.ndarray, discovery: np.ndarray, refined: np.ndarray,
    support: tuple[int, int] | None, box: BoundingBox | None,
) -> list[dict[str, Any]]:
    """Mede suporte e extensão dos componentes sem alterar a máscara."""
    labels, count = mask_components(data)
    records = []
    for index in range(1, count + 1):
        component = labels == index
        y, x = np.nonzero(component)
        records.append({
            "component_id": index,
            "area": len(x),
            "box": mask_box(component),
            "contains_support": support is not None and bool(component[support[1], support[0]]),
            "discovery_overlap": int((component & discovery).sum()),
            "refined_overlap": int((component & refined).sum()),
            "detector_box_overlap": None if box is None else int((component & box_mask(box, data.shape)).sum()),
            "distance_to_box_center_px": None if box is None else float(np.sqrt(
                (x - (box.x_min + box.x_max) / 2) ** 2 + (y - (box.y_min + box.y_max) / 2) ** 2
            ).min()),
        })
    return records


# Mantém o componente ancorado para thing/part/unknown. Stuff é suporte de
# superfície e pode ser descontínuo por oclusão, sem exigir identidade única.
def _supported_component(
    data: np.ndarray, kind: RegionKind, support: tuple[int, int] | None,
) -> tuple[np.ndarray, bool]:
    """Retorna pixels autorizados e se o suporte principal sobreviveu."""
    if kind is RegionKind.STUFF:
        return data.copy(), bool(data.any())
    labels, _ = mask_components(data)
    if support is None or not data[support[1], support[0]]:
        return np.zeros_like(data), False
    return labels == labels[support[1], support[0]], True


# Converte a saída de segmentação em footprint semântico, preservando todos os
# estágios geométricos no registro de grounding da região original.
def _validate_prediction(
    region: ObservedRegion, prediction: GroundingPrediction,
    usable: np.ndarray, config: SemanticGroundingConfig,
) -> SemanticGrounding:
    """Valida conceito, frame, clipping e conectividade de uma predição."""
    claim = primary_label_claim(region)
    assert claim is not None
    diagnostics: dict[str, Any] = {
        "region_id": region.region_id, "label": claim.value,
        "region_kind": (claim.region_kind or RegionKind.UNKNOWN).value,
        "discovery_area": region.mask.area(), "box_before": asdict(region.box),
        "component_count_before": mask_components(region.mask.data)[1],
        "geometric_confidence": region.geometric_confidence,
        "visual_support": None if claim.support is None else claim.support.visual_support,
        "model_calls": prediction.model_calls, "latency_s": prediction.latency_s,
    }
    status = prediction.status
    accepted = np.zeros_like(region.mask.data)
    support = None
    if prediction.region_id != region.region_id or prediction.concept != claim.value:
        raise ValueError("Grounder returned a different region or concept.")
    if prediction.model_mask is not None:
        if prediction.model_mask.data.shape != region.mask.data.shape:
            diagnostics["reason"] = "A resolução da segmentação difere da imagem original."
            status = GroundingStatus.INCONSISTENT
            prediction = replace(prediction, model_mask=None, status=status)
        elif status is GroundingStatus.REFINED:
            assert prediction.prompt_box is not None
            assert prediction.model_mask is not None
            raw = prediction.model_mask.data
            localization = np.zeros_like(raw)
            for box in prediction.prompt_boxes or (prediction.prompt_box,):
                localization |= box_mask(box, raw.shape)
            localized = raw & localization
            support = _principal_support(localized, prediction.prompt_box)
            clipped = localized & region.mask.data & usable
            diagnostics["model_mask_area"] = int(raw.sum())
            diagnostics["clipped_pixel_count"] = int(raw.sum() - clipped.sum())
            diagnostics["components"] = _component_diagnostics(
                clipped, region.mask.data, raw, support, prediction.prompt_box
            )
            accepted, anchored = _supported_component(clipped, claim.region_kind or RegionKind.UNKNOWN, support)
            if not clipped.any():
                status = GroundingStatus.INCONSISTENT
            elif not anchored:
                status = GroundingStatus.FRAGMENTED
            elif int(accepted.sum()) < config.min_mask_area:
                accepted[:] = False
                status = GroundingStatus.TOO_SMALL
    diagnostics.update({
        "semantic_grounding_status": status.value,
        "semantic_area": int(accepted.sum()),
        "area_ratio": float(accepted.sum()) / region.mask.area() if region.mask.area() else 0,
        "area_reduction_ratio": 1 - float(accepted.sum()) / region.mask.area() if region.mask.area() else 0,
        "component_count_after": mask_components(accepted)[1],
        "box_after": mask_box(accepted),
        "reason": diagnostics.get("reason", prediction.reason),
    })
    return SemanticGrounding(
        prediction, status,
        Mask(accepted, region.mask.image_width, region.mask.image_height) if accepted.any() else None,
        support, diagnostics,
    )


# Executa somente as candidatas públicas, permitindo que o backend compartilhe
# trabalho por frame. Falhas mantêm claims e tornam ausência espacial explícita.
def ground_regions(
    regions: tuple[ObservedRegion, ...], image: ImagePayload,
    grounder: SemanticGrounder | None, config: SemanticGroundingConfig,
    area_masks: ImageAreaMasks | None = None,
) -> tuple[ObservedRegion, ...]:
    """Anexa grounding às regiões sem modificar discovery, claims ou embeddings."""
    usable = np.ones((image.height, image.width), dtype=np.bool_)
    if len({region.region_id for region in regions}) != len(regions):
        raise ValueError("Grounding regions must have unique identities.")
    if any(region.mask.data.shape != usable.shape for region in regions):
        raise ValueError("Grounding discovery dimensions must match the source image.")
    if area_masks is not None:
        if any(mask is not None and mask.data.shape != usable.shape for mask in (area_masks.valid_area, area_masks.ego_vehicle)):
            raise ValueError("Grounding area masks must match the source image.")
        if area_masks.valid_area is not None:
            usable &= area_masks.valid_area.data
        if area_masks.ego_vehicle is not None:
            usable &= ~area_masks.ego_vehicle.data
    requests = tuple(
        GroundingRequest(region.region_id, claim.value, claim.region_kind or RegionKind.UNKNOWN, region.mask)
        for region in regions if (claim := primary_label_claim(region)) is not None
    )
    predictions: tuple[GroundingPrediction, ...] = ()
    reason = "Nenhum backend de grounding foi configurado."
    status = GroundingStatus.UNAVAILABLE
    if grounder is not None and requests:
        try:
            predictions = grounder.ground(image, requests, config)
        except (OSError, BackendUnavailableError) as error:
            reason, status = str(error), GroundingStatus.UNAVAILABLE
        except (VisualPerceptionError, RuntimeError, ValueError) as error:
            reason, status = str(error), GroundingStatus.FAILED
        else:
            reason, status = "O backend omitiu a predição solicitada.", GroundingStatus.FAILED
    by_id = {item.region_id: item for item in predictions}
    if len(by_id) != len(predictions) or set(by_id) - {item.region_id for item in requests}:
        raise ValueError("Grounding predictions must have unique requested region ids.")
    updated = []
    for region in regions:
        claim = primary_label_claim(region)
        if claim is None:
            updated.append(region)
            continue
        prediction = by_id.get(region.region_id) or GroundingPrediction(
            region.region_id, claim.value, reason=reason, status=status
        )
        grounding = _validate_prediction(region, prediction, usable, config)
        updated.append(replace(region, grounding=grounding))
    return tuple(updated)


# Resolve a sobreposição no módulo dono das máscaras. Consumidores recebem
# tanto regiões aceitas quanto abstenções, e não precisam reinventar ownership.
def build_spatial_footprints(
    regions: tuple[ObservedRegion, ...], usable_mask: Mask,
    *, stuff_discovery_fallback: bool = False, legacy_discovery: bool = False,
) -> tuple[SpatialRegionFootprint, ...]:
    """Produz footprints disjuntos e auditáveis após ownership.

    ``legacy_discovery`` é somente o braço explícito de comparação histórica.
    O default nunca converte um artifact antigo em grounding.
    """
    candidates = []
    for region in regions:
        if region.mask.data.shape != usable_mask.data.shape:
            raise ValueError("Footprint and usable mask dimensions must agree.")
        claim = primary_label_claim(region)
        if claim is None:
            continue
        grounding = region.grounding
        strong = grounding is not None and grounding.status is GroundingStatus.REFINED
        selected = grounding.semantic_mask if grounding is not None and strong else None
        fallback = stuff_discovery_fallback and claim.region_kind is RegionKind.STUFF
        if legacy_discovery or (selected is None and fallback):
            selected = region.mask
        mask = np.zeros_like(usable_mask.data) if selected is None else selected.data & usable_mask.data
        diagnostics: dict[str, Any] = dict(grounding.diagnostics) if grounding is not None else {
            "region_id": region.region_id, "label": claim.value,
            "region_kind": (claim.region_kind or RegionKind.UNKNOWN).value,
            "discovery_area": region.mask.area(), "semantic_area": 0,
            "area_ratio": 0.0, "area_reduction_ratio": 1.0,
            "component_count_before": mask_components(region.mask.data)[1], "component_count_after": 0,
            "box_before": asdict(region.box), "box_after": None,
            "geometric_confidence": region.geometric_confidence,
            "visual_support": None if claim.support is None else claim.support.visual_support,
            "semantic_grounding_status": GroundingStatus.UNAVAILABLE.value,
        }
        diagnostics.update({"legacy_discovery": legacy_discovery, "stuff_tentative_fallback": fallback and not strong})
        candidates.append((not (strong or legacy_discovery), int(mask.sum()), region, claim, mask, diagnostics))
    ownership = np.zeros_like(usable_mask.data)
    footprints = []
    for tentative, _, region, claim, mask, diagnostics in sorted(candidates, key=lambda item: (
        item[0], item[1], -(item[3].confidence.value if item[3].confidence is not None else 0.0), item[2].region_id,
    )):
        residual = mask & ~ownership
        grounding = region.grounding
        support = None if grounding is None else grounding.support_pixel
        box = None if grounding is None else grounding.prediction.prompt_box
        diagnostics["components_after_ownership"] = _component_diagnostics(residual, region.mask.data, mask, support, box)
        diagnostics["component_count_before_ownership"] = mask_components(mask)[1]
        diagnostics["component_count_after_ownership"] = mask_components(residual)[1]
        accepted = residual
        if not legacy_discovery and not tentative and mask.any():
            accepted, anchored = _supported_component(residual, claim.region_kind or RegionKind.UNKNOWN, support)
            if not anchored:
                diagnostics["semantic_grounding_status"] = GroundingStatus.FRAGMENTED.value
        diagnostics["ownership_removed_pixel_count"] = int(mask.sum() - residual.sum())
        diagnostics["fragment_removed_pixel_count"] = int(residual.sum() - accepted.sum())
        diagnostics["association_area"] = int(accepted.sum())
        diagnostics["component_count_final"] = mask_components(accepted)[1]
        diagnostics["association_box"] = mask_box(accepted)
        ownership |= accepted
        footprints.append(SpatialRegionFootprint(
            region.region_id, claim.value,
            Mask(accepted, usable_mask.image_width, usable_mask.image_height) if accepted.any() else None,
            not tentative and bool(accepted.any()), diagnostics,
        ))
    return tuple(footprints)
