"""Mapeia os tipos de domínio para formas desenháveis e compõe os overlays.

Usado por ``validate_reference_pipeline.py`` (#190) para gerar as amostras que
acompanham a validação do pipeline real. Não é parte do contract público do
módulo — é uma ferramenta de inspeção local.

A separação em relação a ``render_layers.py`` é deliberada: lá ficam as
primitivas que só entendem geometria; aqui fica o conhecimento de
``ObservedRegion`` e ``RegionProposal``. É o que permite desenhar os dois
estágios com o mesmo código sem que as primitivas conheçam o domínio.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from render_layers import DrawableShape, blend_masks, draw_boxes, draw_labels
from visual_perception.domain.regions import ObservedRegion, RegionProposal, primary_label_claim
from visual_perception.domain.visual_observation import VisualObservation


# Converte regiões observadas em formas desenháveis, carregando os dois eixos
# de confiança separadamente. A confiança semântica vem do claim de label
# primário — a mesma política que o diagnóstico usa, via primary_label_claim,
# para que overlay e diagnostics.json nunca contem labels diferentes.
def region_shapes(regions: Sequence[ObservedRegion]) -> tuple[DrawableShape, ...]:
    """Retorna as formas desenháveis das regiões, com label e confianças."""
    shapes: list[DrawableShape] = []
    for region in regions:
        claim = primary_label_claim(region)
        confidence = None if claim is None or claim.confidence is None else claim.confidence.value
        shapes.append(
            DrawableShape(
                shape_id=region.region_id,
                mask=region.mask,
                box=region.box,
                label=None if claim is None else claim.value,
                semantic_confidence=confidence,
                geometric_confidence=region.geometric_confidence,
            )
        )
    return tuple(shapes)


# Converte proposals de discovery em formas desenháveis. Label e confiança
# semântica ficam em None de propósito: proposals são geometria pura, anteriores
# a qualquer interpretação, e o artifact precisa mostrar essa ausência em vez de
# sugerir uma semântica que aquele estágio não produziu.
def proposal_shapes(proposals: Sequence[RegionProposal]) -> tuple[DrawableShape, ...]:
    """Retorna as formas desenháveis das proposals, sem semântica."""
    return tuple(
        DrawableShape(
            shape_id=proposal.proposal_id,
            mask=proposal.mask,
            box=proposal.box,
            geometric_confidence=proposal.geometric_confidence,
        )
        for proposal in proposals
    )


# Desenha máscaras, caixas e rótulos de uma observação sobre a imagem original.
# Mantida com a assinatura de sempre para os consumidores existentes, mas agora
# composta das primitivas — o artifact completo é uma composição das camadas,
# não uma implementação paralela a elas.
def render_overlay(image: Image.Image, observation: VisualObservation) -> Image.Image:
    """Retorna uma cópia de ``image`` com as regiões da observação desenhadas por cima."""
    shapes = region_shapes(observation.regions)
    return draw_labels(draw_boxes(blend_masks(image, shapes), shapes), shapes)
