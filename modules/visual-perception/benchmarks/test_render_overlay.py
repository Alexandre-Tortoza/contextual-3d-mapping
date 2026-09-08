"""Testes do renderer de inspeção (#212).

O bug que estes testes existem para impedir de voltar: o overlay imprimia
``label (0.97)``, onde ``0.97`` era a ``geometric_confidence`` da máscara. Quem
lia entendia "97% de certeza que isso é uma parede", quando o número não dizia
nada sobre o label. O caveat está registrado em ``docs/known-limitations.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent / "tests"))

from PIL import Image  # noqa: E402

from fixtures import image_observation  # noqa: E402
from render_layers import DrawableShape, confidence_caption, label_caption  # noqa: E402
from render_overlay import proposal_shapes, region_shapes, render_overlay  # noqa: E402
from visual_perception.domain.geometry import Mask  # noqa: E402
from visual_perception.domain.references import ModelProvenance  # noqa: E402
from visual_perception.domain.regions import ObservedRegion, RegionProposal, TileProvenance  # noqa: E402
from visual_perception.domain.semantics import (  # noqa: E402
    ClaimKind,
    ConfidenceScore,
    Evidence,
    HypothesisRole,
    SemanticClaim,
)
from visual_perception.domain.visual_observation import SceneContext, VisualObservation  # noqa: E402

# Grande o bastante para o rótulo de duas linhas caber sem cobrir o frame
# inteiro: em um canvas menor que o texto, duas renderizações diferentes
# resultariam na mesma imagem toda preta e o teste de regressão não mediria nada.
_WIDTH = 220
_HEIGHT = 160


# Constrói uma região com um label de confiança semântica controlada, para os
# testes que precisam variar um eixo de confiança de cada vez.
def _region(*, semantic: float | None, geometric: float) -> ObservedRegion:
    data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    data[40:120, 40:180] = True
    mask = Mask(data, _WIDTH, _HEIGHT)
    claim = SemanticClaim(
        ClaimKind.LABEL,
        "wall",
        None if semantic is None else ConfidenceScore(semantic, "fake"),
        (Evidence(description="raw multimodal region response"),),
        ModelProvenance(stage="region_semantics", producer="fake", config_fingerprint="fp"),
        role=HypothesisRole.PRIMARY,
    )
    return ObservedRegion("region-a", mask, mask.bounding_box(), geometric, ("region-a-p",), (claim,))


# Monta uma observação de uma região só, suficiente para renderizar.
def _observation(region: ObservedRegion) -> VisualObservation:
    return VisualObservation(
        source=image_observation(width=_WIDTH, height=_HEIGHT).source,
        image_width=_WIDTH,
        image_height=_HEIGHT,
        scene_context=SceneContext(),
        regions=(region,),
        relations=(),
    )


# Uma base neutra para as comparações pixel a pixel entre renderizações.
def _base_image() -> Image.Image:
    return Image.fromarray(np.full((_HEIGHT, _WIDTH, 3), 255, dtype=np.uint8), mode="RGB")


# Os dois eixos precisam aparecer nomeados. Sem os nomes, qualquer número ao
# lado de um label volta a ser lido como certeza semântica.
def test_caption_names_both_confidence_axes() -> None:
    """A legenda identifica explicitamente qual número é qual."""
    assert confidence_caption(0.90, 0.97) == "sem=0.90 geom=0.97"


# Ausência de score é distinta de score baixo, e o artifact precisa mostrar essa
# diferença: o backend pode omitir a confiança, e o contract permite ``None``.
def test_caption_marks_an_absent_semantic_score_as_unknown() -> None:
    """Sem confiança semântica, a legenda mostra ``sem=?`` em vez de um número."""
    assert confidence_caption(None, 0.97) == "sem=? geom=0.97"


# Proposals não têm nenhum dos dois eixos garantidos; a função precisa ser total
# para que desenhar o estágio pré-merge nunca estoure.
def test_caption_handles_a_shape_with_no_confidence_at_all() -> None:
    """Sem nenhum dos dois scores, a legenda ainda é formada."""
    assert confidence_caption(None, None) == "sem=? geom=?"


# Asserção negativa contra o formato antigo: qualquer rótulo precisa carregar os
# dois eixos rotulados, nunca um número solto entre parênteses.
def test_label_caption_always_carries_both_named_axes() -> None:
    """O rótulo nunca volta ao formato ``label (0.97)``."""
    caption = label_caption(
        DrawableShape(
            shape_id="s",
            mask=Mask(np.ones((_HEIGHT, _WIDTH), dtype=np.bool_), _WIDTH, _HEIGHT),
            box=Mask(np.ones((_HEIGHT, _WIDTH), dtype=np.bool_), _WIDTH, _HEIGHT).bounding_box(),
            label="wall",
            semantic_confidence=0.42,
            geometric_confidence=0.97,
        )
    )
    assert "sem=0.42" in caption
    assert "geom=0.97" in caption
    assert "(0.97)" not in caption


# O teste de regressão que importa. Um teste só do formatador não pegaria o
# renderer voltar a ignorar a confiança semântica: é preciso provar que variar
# *só* o eixo semântico muda a imagem produzida.
def test_changing_only_the_semantic_confidence_changes_the_rendered_image() -> None:
    """Duas observações que diferem só na confiança do label renderizam diferente."""
    low = render_overlay(_base_image(), _observation(_region(semantic=0.10, geometric=0.97)))
    high = render_overlay(_base_image(), _observation(_region(semantic=0.95, geometric=0.97)))

    assert np.array(low).tobytes() != np.array(high).tobytes()


# O espelho do anterior: a confiança geométrica continua sendo mostrada, e
# continua sendo um eixo próprio. Corrigir a leitura não pode ter custado o dado.
def test_changing_only_the_geometric_confidence_changes_the_rendered_image() -> None:
    """Duas observações que diferem só na confiança geométrica renderizam diferente."""
    low = render_overlay(_base_image(), _observation(_region(semantic=0.90, geometric=0.20)))
    high = render_overlay(_base_image(), _observation(_region(semantic=0.90, geometric=0.99)))

    assert np.array(low).tobytes() != np.array(high).tobytes()


# Uma região sem score de confiança precisa continuar renderizável: a ausência
# é um estado válido do contract, não um erro.
def test_region_without_semantic_score_still_renders() -> None:
    """Uma região com ``confidence=None`` é renderizada, marcada como desconhecida."""
    shapes = region_shapes((_region(semantic=None, geometric=0.97),))

    assert shapes[0].semantic_confidence is None
    assert "sem=?" in label_caption(shapes[0])
    render_overlay(_base_image(), _observation(_region(semantic=None, geometric=0.97)))


# Proposals são geometria pura, anteriores a qualquer interpretação. Desenhar um
# label nelas sugeriria uma semântica que aquele estágio não produziu.
def test_proposals_carry_no_semantics() -> None:
    """As formas de proposals não têm label nem confiança semântica."""
    data = np.zeros((_HEIGHT, _WIDTH), dtype=np.bool_)
    data[20:80, 20:80] = True
    mask = Mask(data, _WIDTH, _HEIGHT)
    proposal = RegionProposal(
        "sam-0", mask, mask.bounding_box(), 0.88, "sam:test", TileProvenance("scale-0", "tile-0")
    )

    shapes = proposal_shapes((proposal,))

    assert shapes[0].label is None
    assert shapes[0].semantic_confidence is None
    assert shapes[0].geometric_confidence == 0.88
