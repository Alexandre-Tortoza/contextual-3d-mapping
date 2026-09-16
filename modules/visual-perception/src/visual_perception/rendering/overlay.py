"""Mapeia os tipos de domínio para formas desenháveis e compõe os overlays.

Usado por ``debug_artifacts.write_stage_debug_images`` e pelo layout completo
de ``benchmarks/frame_artifacts.py`` (#190) para gerar as camadas de inspeção
da validação do pipeline real. Vive em ``src/`` porque é dependência da API
pública de imagens de debug do módulo, mas continua sendo ferramenta de
renderização, não uma capacidade do domínio.

A separação em relação a ``visual_perception.rendering.layers`` é deliberada:
lá ficam as primitivas que só entendem geometria; aqui fica o conhecimento de
``ObservedRegion`` e ``RegionProposal``. É o que permite desenhar os dois
estágios com o mesmo código sem que as primitivas conheçam o domínio.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from visual_perception.rendering.layers import DrawableShape, blend_masks, draw_boxes, draw_labels
from visual_perception.domain.contextual_evidence import contextual_evidence_claim
from visual_perception.domain.regions import ObservedRegion, RegionProposal, primary_label_claim
from visual_perception.domain.visual_observation import VisualObservation


# Converte regiões observadas em formas desenháveis, carregando os dois eixos
# de confiança separadamente.
#
# O label desenhado é o da hipótese que **carrega** a evidência contextual, e
# só cai para o primário quando não existe nenhuma. A distinção importa: uma
# região publicada porque o refinamento a reinterpretou como ``red wall``
# carrega ``wall`` como primeira hipótese afirmada, e desenhar a primeira faria
# o overlay voltar a exibir exatamente o label que a política tira da frente.
# Uma região de contexto estrutural não tem hipótese portadora, e por isso a
# camada esmaecida continua mostrando o label do produtor.
#
# ``primary_label_claim`` segue sendo a política única de *quem conta labels* —
# ela responde "o que o produtor elegeu", que é outra pergunta e continua
# comparável entre runs em ``diagnostics.json``.
def region_shapes(regions: Sequence[ObservedRegion]) -> tuple[DrawableShape, ...]:
    """Retorna as formas desenháveis das regiões, com label e confianças."""
    shapes: list[DrawableShape] = []
    for region in regions:
        claim = contextual_evidence_claim(region) or primary_label_claim(region)
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


# Converte em formas as superfícies que a política de publicação manteve fora
# do output público. Existe para que a supressão seja inspecionável: o overlay
# principal mostra o que o módulo publica, e esta camada mostra o que ele
# decidiu não publicar — sem ela, "sumiu" e "nunca foi observado" ficariam
# indistinguíveis no artifact.
def structural_context_shapes(observation: VisualObservation) -> tuple[DrawableShape, ...]:
    """Retorna as formas das regiões mantidas como contexto estrutural."""
    return region_shapes(observation.structural_context)


# Desenha máscaras, caixas e rótulos de uma observação sobre a imagem original.
# Mantida com a assinatura de sempre para os consumidores existentes, mas agora
# composta das primitivas — o artifact completo é uma composição das camadas,
# não uma implementação paralela a elas.
#
# Desenha ``observation.regions``, que desde a política de publicação contextual
# contém apenas evidência contextual: uma cena inteiramente normal produz um
# overlay quase vazio, e isso é o comportamento correto.
def render_overlay(image: Image.Image, observation: VisualObservation) -> Image.Image:
    """Retorna uma cópia de ``image`` com as regiões da observação desenhadas por cima."""
    shapes = region_shapes(observation.regions)
    return draw_labels(draw_boxes(blend_masks(image, shapes), shapes), shapes)
