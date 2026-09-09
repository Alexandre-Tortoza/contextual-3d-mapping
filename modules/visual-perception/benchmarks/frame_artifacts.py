"""Persistência dos artifacts de inspeção de um frame validado.

Dono único do layout em disco de um frame. Só escreve: não chama modelo, não
decide configuração e não conhece o rig ou o dataset de origem — quem chama
passa os pixels e as máscaras já resolvidos.

O layout separa uma camada por pergunta, porque um overlay único com dezenas de
caixas não distingue "o SAM propôs errado" de "o merge deveria ter unido" de "a
máscara está certa e só a caixa parece larga":

```text
frames/<frame-id>/
    observation.json      diagnostics.json
    embeddings.npz        os vetores por região, referenciados por artifact_ref
    raw.png               os pixels de origem, nunca modificados
    proposals.png         masks + boxes antes do merge, sem semântica
    regions-masks.png     só as masks finais
    regions-boxes.png     só as boxes finais
    regions-labels.png    labels no centróide, sem caixas
    regions-overlay.png   tudo junto
    ego-mask.png          máscara de exclusão, quando existe
    pipeline-input.png    só quando difere de raw.png
```

As views por região (foreground, tight, contextual) não são persistidas aqui:
elas são função pura da observação, da imagem e da config, todas já salvas, e
``inspect_region.py`` as materializa sob demanda para a região investigada.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from render_layers import binary_mask_image, blend_masks, draw_boxes, draw_labels
from render_overlay import proposal_shapes, region_shapes
from visual_perception.application.observation_diagnostics import ObservationDiagnostics
from visual_perception.application.pipeline import PipelineResult
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.infrastructure.serialization import serialize_observation

#: Versão do layout de artifacts por frame, gravada no manifest. Um leitor que
#: espera outro layout falha explicitamente em vez de ler o diretório errado em
#: silêncio. A ``frames/2`` acrescenta ``embeddings.npz``: até a #217 os vetores
#: eram calculados e descartados dentro do pipeline, e o que ficava no artifact
#: era só a string de ``artifact_ref``, apontando para nada.
FRAME_ARTIFACT_LAYOUT_VERSION = "frames/2"


# Agrupa os pixels distinguíveis de um frame e as máscaras de exclusão do rig.
# Existe para separar, de forma auditável, o que veio do dataset do que o
# pipeline efetivamente recebeu: se os dois divergirem, algum pré-processamento
# alterou a entrada, e isso precisa ser visível em vez de deduzido.
@dataclass(frozen=True)
class FrameInputs:
    """Os pixels de um frame antes e depois do pré-processamento, mais as máscaras.

    Argumentos:
        raw: os pixels exatamente como vieram do dataset.
        pipeline_input: os pixels que o pipeline recebeu.
        ego_mask: máscara do que o rig ocupa na imagem, ou ``None``.
        valid_area_mask: máscara da área útil do sensor, ou ``None``.
    """

    raw: ImagePayload
    pipeline_input: ImagePayload
    ego_mask: np.ndarray | None = None
    valid_area_mask: np.ndarray | None = None

    # Responde se a entrada do pipeline é idêntica ao frame de origem. É a
    # propriedade que o artifact precisa afirmar explicitamente: sem ela, um
    # pré-processamento destrutivo passa despercebido porque o overlay é
    # desenhado sobre a mesma imagem já alterada.
    def pipeline_input_is_raw(self) -> bool:
        """Retorna ``True`` quando o pipeline recebeu exatamente os pixels de origem."""
        return bool(np.array_equal(self.raw.pixels, self.pipeline_input.pixels))


# Converte um payload de pixels em imagem PIL, sem depender do runtime de
# inferência (que exige config de backend). Local porque é detalhe de escrita
# de artifact, não uma capacidade do módulo.
def _to_image(payload: ImagePayload) -> Image.Image:
    """Retorna o payload como imagem RGB."""
    return Image.fromarray(payload.pixels.astype(np.uint8), mode="RGB")


# Escreve os artifacts de um frame e devolve o mapa de nome lógico para caminho
# relativo. O mapa vai para o manifest, de modo que um leitor encontre cada
# camada pelo nome em vez de reconstruir convenções de caminho.
def write_frame_artifacts(
    frame_dir: Path,
    *,
    inputs: FrameInputs,
    result: PipelineResult,
    diagnostics: ObservationDiagnostics,
    extra_diagnostics: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Persiste as camadas de inspeção de um frame e retorna seus caminhos relativos.

    Argumentos:
        frame_dir: diretório do frame; criado se não existir.
        inputs: pixels de origem, pixels entregues ao pipeline e máscaras.
        result: resultado do pipeline, incluindo as proposals pré-merge.
        diagnostics: o resumo estatístico já calculado para a observação.
        extra_diagnostics: campos adicionais do harness (identidade do frame,
            hash da entrada, latência) mesclados no ``diagnostics.json``.
    Retorna:
        mapa de nome lógico do artifact para o caminho relativo a ``frame_dir``.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # Um write_text por artifact, com o nome lógico registrado no mesmo passo,
    # para que o manifest nunca liste um caminho que não foi escrito.
    def _record(name: str, path: Path) -> None:
        written[name] = str(path.relative_to(frame_dir))

    observation_path = frame_dir / "observation.json"
    observation_path.write_text(json.dumps(serialize_observation(result.observation), indent=2))
    _record("observation", observation_path)

    # Os vetores viajam **por referência**: a observação guarda o
    # ``artifact_ref`` de cada slot, e o arquivo abaixo é o artifact que aquela
    # referência resolve. Embutí-los no JSON inflaria a observação canônica em
    # duas ordens de grandeza sem tornar nada mais auditável.
    embeddings = {
        embedding.embedding_id: np.asarray(embedding.vector, dtype=np.float32)
        for embedding in (*result.visual_embeddings, *result.language_embeddings)
    }
    if embeddings:
        embeddings_path = frame_dir / "embeddings.npz"
        np.savez_compressed(embeddings_path, **embeddings)
        _record("embeddings", embeddings_path)

    raw_image = _to_image(inputs.raw)
    raw_path = frame_dir / "raw.png"
    raw_image.save(raw_path)
    _record("raw", raw_path)

    # Escrito apenas quando difere: dois PNGs idênticos por frame não informam
    # nada, e a igualdade fica afirmada no diagnóstico, não por comparação
    # manual de arquivos.
    if not inputs.pipeline_input_is_raw():
        pipeline_input_path = frame_dir / "pipeline-input.png"
        _to_image(inputs.pipeline_input).save(pipeline_input_path)
        _record("pipeline_input", pipeline_input_path)

    if inputs.valid_area_mask is not None:
        valid_path = frame_dir / "valid-area-mask.png"
        binary_mask_image(inputs.valid_area_mask).save(valid_path)
        _record("valid_area_mask", valid_path)
    if inputs.ego_mask is not None:
        ego_path = frame_dir / "ego-mask.png"
        binary_mask_image(inputs.ego_mask).save(ego_path)
        _record("ego_mask", ego_path)

    # Proposals nunca recebem rótulo: elas são anteriores a qualquer
    # interpretação, e desenhar um label aqui sugeriria uma semântica que este
    # estágio não produziu.
    proposals = proposal_shapes(result.proposals)
    proposals_path = frame_dir / "proposals.png"
    draw_boxes(blend_masks(raw_image, proposals), proposals).save(proposals_path)
    _record("proposals", proposals_path)

    regions = region_shapes(result.observation.regions)
    masks_path = frame_dir / "regions-masks.png"
    blend_masks(raw_image, regions).save(masks_path)
    _record("regions_masks", masks_path)

    boxes_path = frame_dir / "regions-boxes.png"
    draw_boxes(raw_image, regions).save(boxes_path)
    _record("regions_boxes", boxes_path)

    labels_path = frame_dir / "regions-labels.png"
    draw_labels(blend_masks(raw_image, regions), regions).save(labels_path)
    _record("regions_labels", labels_path)

    overlay_path = frame_dir / "regions-overlay.png"
    draw_labels(draw_boxes(blend_masks(raw_image, regions), regions), regions).save(overlay_path)
    _record("regions_overlay", overlay_path)

    payload: dict[str, object] = {
        "layout_version": FRAME_ARTIFACT_LAYOUT_VERSION,
        **dict(extra_diagnostics or {}),
        "pipeline_input_identical_to_raw": inputs.pipeline_input_is_raw(),
        # Estes dois campos descrevem *exclusão aplicada*, e não pixels
        # pintados: a geometria vem declarada da sequência e é aplicada na
        # filtragem de proposals. ``diagnostics.ego`` e ``diagnostics.fisheye``
        # trazem as contagens; estes resumem se havia geometria declarada.
        "ego_vehicle_mask_applied": diagnostics.ego.applied,
        "valid_fisheye_mask": "applied" if diagnostics.fisheye.applied else "unavailable",
        **asdict(diagnostics),
    }
    diagnostics_path = frame_dir / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(payload, indent=2))
    _record("diagnostics", diagnostics_path)

    return written
