"""Refinamento seletivo guiado por incerteza.

Issue: #183.

Apenas regions sinalizadas por sinais de incerteza estruturados são
reprocessadas. Evidência anterior nunca é sobrescrita: o refinamento
adiciona (append) novas claims (o mesmo comportamento append-only que a
interpretação de region já tem, ver #165) e cada iteração é registrada
no histórico retornado. O loop sempre termina: para assim que nenhuma
region ainda precisa de refinamento, e nunca executa mais que
``RefinementConfig.max_iterations`` passes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from visual_perception.application.quality_audit import audit_observation
from visual_perception.application.region_semantics import interpret_regions
from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.errors import RegionInterpretationFailure
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.visual_observation import VisualObservation
from visual_perception.ports.multimodal_reasoning import MultimodalReasoner


# Configuração dos thresholds que decidem quais regions são reprocessadas
# pelo loop de refinamento (confiança baixa, region pequena demais, número
# máximo de iterações). Existe para deixar esses limiares explícitos e
# validados em vez de constantes espalhadas pelo código de refine_observation.
@dataclass(frozen=True)
class RefinementConfig:
    low_confidence_threshold: float = 0.5
    small_region_area_px: int = 16
    max_iterations: int = 2

    # Valida os invariantes da configuração logo após a construção da
    # dataclass congelada (frozen), para falhar cedo com um erro acionável
    # em vez de propagar thresholds inválidos para o loop de refinamento.
    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be in [0, 1].")
        if self.small_region_area_px < 0:
            raise ValueError("small_region_area_px must be non-negative.")
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative.")


# Registro de uma iteração de refinamento: quais regions foram reprocessadas
# e por quê. Existe para compor o histórico retornado por refine_observation,
# permitindo auditar depois o que mudou e o que falhou em cada passe.
@dataclass(frozen=True)
class RefinementStep:
    """Registro de uma iteração de refinamento: quais regions foram reprocessadas e por quê."""

    iteration: int
    target_region_ids: tuple[str, ...]
    failures: tuple[RegionInterpretationFailure, ...]


# Decide quais regions ainda precisam de um novo passe de interpretação,
# combinando o resultado do audit de qualidade (claims contraditórias) com
# heurísticas locais (region pequena, confiança baixa, relations incertas).
# Chamada por refine_observation a cada iteração do loop de refinamento.
def select_refinement_targets(observation: VisualObservation, config: RefinementConfig) -> tuple[str, ...]:
    """Seleciona os ids de region que precisam de outro passe de interpretação."""
    audit = audit_observation(observation)
    targets = {
        issue.region_id
        for issue in audit.warnings
        if issue.code == "contradictory_claims" and issue.region_id is not None
    }
    for region in observation.regions:
        too_small = region.mask.area() < config.small_region_area_px
        low_confidence = any(
            claim.confidence.value < config.low_confidence_threshold for claim in region.claims
        )
        if not region.claims or too_small or low_confidence:
            targets.add(region.region_id)
    for relation in observation.relations:
        if relation.confidence.value < config.low_confidence_threshold:
            targets.add(relation.subject_region_id)
            targets.add(relation.object_region_id)
    return tuple(sorted(targets))


# Ponto de entrada do refinamento: reprocessa seletivamente as regions
# incertas de uma VisualObservation até o loop convergir (nenhum target
# restante) ou atingir max_iterations. Existe para melhorar a qualidade da
# observation sem reprocessar tudo, custeando reasoning só onde há incerteza
# real (issue #183); usada pelo pipeline de percepção como passo opcional
# pós-interpretação.
def refine_observation(
    observation: VisualObservation,
    image: ImagePayload,
    reasoner: MultimodalReasoner,
    multimodal_config: MultimodalReasoningConfig,
    refinement_config: RefinementConfig,
) -> tuple[VisualObservation, tuple[RefinementStep, ...]]:
    """Reprocessa seletivamente as regions incertas até o loop convergir ou parar."""
    current = observation
    history: list[RefinementStep] = []

    for iteration in range(refinement_config.max_iterations):
        target_ids = select_refinement_targets(current, refinement_config)
        if not target_ids:
            break

        order = {region.region_id: index for index, region in enumerate(current.regions)}
        targets = tuple(region for region in current.regions if region.region_id in target_ids)
        others = tuple(region for region in current.regions if region.region_id not in target_ids)

        refined_targets, failures = interpret_regions(
            targets, image, current.scene_context, reasoner, multimodal_config
        )
        updated_regions = tuple(
            sorted(others + refined_targets, key=lambda region: order[region.region_id])
        )
        current = dataclasses.replace(current, regions=updated_regions)
        history.append(RefinementStep(iteration=iteration, target_region_ids=target_ids, failures=failures))

    return current, tuple(history)
