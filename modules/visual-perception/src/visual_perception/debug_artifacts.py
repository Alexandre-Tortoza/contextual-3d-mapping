"""API pública para escrever as imagens de debug de um frame já processado.

Dono da lógica de desenho que antes vivia só em
``benchmarks/frame_artifacts.py``: máscaras, caixas e rótulos das camadas de
inspeção (raw, proposals, regions-*, semantic-overlay, structural-context),
mais as passadas de discovery e as views por região entregues ao reasoner.
Promovida a ``src/`` porque a geração dessas imagens deixou de ser exclusiva
do harness de benchmark — o pipeline de produção (``mapping-runtime``) também
pode gerá-las, atrás de uma flag opt-in que pertence à camada de chamada, não
a este módulo.

Este módulo não decide *se* as imagens devem ser escritas: quem chama recebe
``config`` explícito e obrigatório, e decide se invoca
``write_stage_debug_images`` ou não. ``benchmarks/frame_artifacts.py``
continua dono do layout completo de um frame de benchmark (observation.json,
embeddings.npz, diagnostics.json, trilha de grounding) e delega a esta função
apenas a parte de desenho.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from visual_perception.application.pipeline import PipelineResult
from visual_perception.application.region_views import build_region_views
from visual_perception.application.tiling import build_tiles
from visual_perception.config import ModuleConfig
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.rendering.layers import DrawableShape, blend_masks, draw_boxes, draw_labels
from visual_perception.rendering.overlay import proposal_shapes, region_shapes, structural_context_shapes


# Agrupa os pixels distinguíveis de um frame e as máscaras de exclusão do rig.
# Existe para separar, de forma auditável, o que veio do dataset do que o
# pipeline efetivamente recebeu: se os dois divergirem, algum pré-processamento
# alterou a entrada, e isso precisa ser visível em vez de deduzido. Pública
# porque é o tipo de entrada de ``write_stage_debug_images``.
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


# Agrupa as imagens de debug escritas para um frame, separadas pela forma como
# um leitor precisa indexá-las: um dicionário plano para as camadas únicas por
# frame, e listas para as camadas que se repetem por tile ou por região.
@dataclass(frozen=True)
class StageDebugImages:
    """Caminhos relativos das imagens de debug escritas para um frame.

    Argumentos:
        images: mapa de nome lógico de camada (``"raw"``, ``"proposals"``,
            ``"regions-masks"``, ``"regions-boxes"``, ``"regions-labels"``,
            ``"regions-overlay"``, e, quando aplicável, ``"semantic-overlay"``
            e ``"structural-context"``) para o caminho relativo a ``frame_dir``.
        discovery_tiles: uma entrada por passada de discovery, com ``id``,
            ``input`` e ``proposals``.
        region_views: uma entrada por região com pelo menos uma view
            materializada, com ``region_id`` e as views presentes entre
            ``masked_subject``, ``tight_crop`` e ``contextual_crop``.
    """

    images: dict[str, str]
    discovery_tiles: tuple[dict[str, str], ...]
    region_views: tuple[dict[str, str], ...]


# Converte um payload de pixels em imagem PIL, sem depender do runtime de
# inferência (que exige config de backend). Local porque é detalhe de escrita
# de artifact, não uma capacidade do módulo.
def _to_image(payload: ImagePayload) -> Image.Image:
    """Retorna o payload como imagem RGB."""
    return Image.fromarray(payload.pixels.astype(np.uint8), mode="RGB")

# Persiste a entrada e o overlay de propostas de cada passada do discovery.
# Existe porque ``proposals.png`` agrega a imagem inteira e escondia se uma
# máscara veio da passagem global ou de um tile; usa as proposals já
# retornadas pelo pipeline, portanto não reexecuta o discoverer nem altera a
# inferência auditada.
def _write_discovery_tile_images(
    frame_dir: Path, image: ImagePayload, config: ModuleConfig, result: PipelineResult,
) -> tuple[dict[str, str], ...]:
    """Grava a entrada e o overlay de propostas de cada passada do discovery.

    Argumentos:
        frame_dir: diretório raiz do frame, usado para relativizar os caminhos.
        image: pixels exatos recebidos pelo pipeline.
        config: configuração que define a malha de tiles.
        result: saída com as propostas brutas anteriores à filtragem.
    Retorna:
        uma entrada ``{"id", "input", "proposals"}`` por passada de discovery.
    """
    discovery_root = frame_dir / "discovery"
    discovery_root.mkdir(parents=True, exist_ok=True)
    shapes = proposal_shapes(result.discovered_proposals)
    global_overlay = draw_boxes(blend_masks(_to_image(image), shapes), shapes)
    tiles: list[dict[str, str]] = []
    for tile in build_tiles(image, config.tiling):
        offset_x = int(tile.transform.offset_x)
        offset_y = int(tile.transform.offset_y)
        bounds = (offset_x, offset_y, offset_x + tile.payload.width, offset_y + tile.payload.height)
        stem = f"{tile.scale_id}-{tile.tile_id}"
        input_path = discovery_root / f"{stem}-input.png"
        proposals_path = discovery_root / f"{stem}-proposals.png"
        _to_image(tile.payload).save(input_path)
        global_overlay.crop(bounds).save(proposals_path)
        tiles.append({
            "id": stem,
            "input": str(input_path.relative_to(frame_dir)),
            "proposals": str(proposals_path.relative_to(frame_dir)),
        })
    return tuple(tiles)


# Materializa as três views entregues ao reasoner. Existe para permitir
# revisão visual de uma run sem reconstruir recortes; é chamada depois da
# observação final, cuja geometria é congelada após merge.
def _write_region_view_images(
    frame_dir: Path, image: ImagePayload, config: ModuleConfig, result: PipelineResult,
) -> tuple[dict[str, str], ...]:
    """Grava masked subject, tight crop e contextual crop por região final.

    Argumentos:
        frame_dir: diretório raiz do frame, usado para relativizar os caminhos.
        image: pixels exatos recebidos pelo pipeline.
        config: configuração que define quais views existem.
        result: observação final usada pelo reasoner e pela publicação.
    Retorna:
        uma entrada ``{"region_id", ...}`` por região com ao menos uma view.
    """
    filenames = {
        "masked_subject": "masked-subject.png",
        "tight_crop": "tight-crop.png",
        "contextual_crop": "contextual-crop.png",
    }
    views_by_region = build_region_views(
        result.observation.all_regions, image, config, area_masks=result.area_masks
    )
    entries: list[dict[str, str]] = []
    for region_id, views in views_by_region.items():
        region_root = frame_dir / "regions" / region_id
        entry: dict[str, str] = {"region_id": region_id}
        for view in views:
            filename = filenames.get(view.slot.value)
            if filename is None:
                continue
            region_root.mkdir(parents=True, exist_ok=True)
            path = region_root / filename
            _to_image(view.payload).save(path)
            entry[view.slot.value] = str(path.relative_to(frame_dir))
        if len(entry) > 1:
            entries.append(entry)
    return tuple(entries)


# Escreve as imagens de debug de um frame já processado e devolve seus
# caminhos relativos a ``frame_dir``. É a fronteira pública que qualquer
# produtor de artifacts do módulo (benchmark ou pipeline de produção) chama
# para materializar as mesmas camadas de inspeção, sem duplicar a lógica de
# desenho. Usada por ``benchmarks/frame_artifacts.py``.
def write_stage_debug_images(
    frame_dir: Path, *, inputs: FrameInputs, result: PipelineResult, config: ModuleConfig,
) -> StageDebugImages:
    """Persiste as camadas de inspeção visual de um frame e retorna seus caminhos.

    Argumentos:
        frame_dir: diretório do frame; criado se não existir.
        inputs: pixels de origem, pixels entregues ao pipeline e máscaras.
        result: resultado do pipeline, incluindo as proposals pré-merge.
        config: configuração efetiva da run, usada para a malha de tiles do
            discovery e para as views por região entregues ao reasoner.
    Retorna:
        as imagens escritas, organizadas por camada, tile de discovery e região.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    def _record(name: str, path: Path) -> None:
        written[name] = str(path.relative_to(frame_dir))

    raw_image = _to_image(inputs.raw)
    raw_path = frame_dir / "raw.png"
    raw_image.save(raw_path)
    _record("raw", raw_path)

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
    _record("regions-masks", masks_path)

    boxes_path = frame_dir / "regions-boxes.png"
    draw_boxes(raw_image, regions).save(boxes_path)
    _record("regions-boxes", boxes_path)

    labels_path = frame_dir / "regions-labels.png"
    draw_labels(blend_masks(raw_image, regions), regions).save(labels_path)
    _record("regions-labels", labels_path)

    overlay_path = frame_dir / "regions-overlay.png"
    draw_labels(draw_boxes(blend_masks(raw_image, regions), regions), regions).save(overlay_path)
    _record("regions-overlay", overlay_path)

    # O preview semântico usa somente a máscara explicitamente grounded. As
    # camadas anteriores continuam mostrando discovery para comparação.
    grounded_shapes = [
        DrawableShape(region.region_id, region.grounding.semantic_mask,
            region.grounding.semantic_mask.bounding_box(), region.grounding.prediction.concept)
        for region in result.observation.regions
        if region.grounding is not None and region.grounding.semantic_mask is not None
    ]
    # Escrito só quando há o que desenhar: sem nenhuma máscara grounded o PNG
    # saía idêntico a raw.png e era registrado como artifact de grounding, e
    # "o grounding falhou" ficava igual a "não havia nada".
    if grounded_shapes:
        semantic_path = frame_dir / "semantic-overlay.png"
        draw_labels(blend_masks(raw_image, grounded_shapes), grounded_shapes).save(semantic_path)
        _record("semantic-overlay", semantic_path)

    # A camada do contexto estrutural usa alpha baixo de propósito: ela existe
    # para responder "o que foi suprimido, e onde", e não para competir
    # visualmente com a evidência que o módulo de fato publica.
    structural = structural_context_shapes(result.observation)
    if structural:
        structural_path = frame_dir / "structural-context.png"
        draw_labels(blend_masks(raw_image, structural, alpha=0.25), structural).save(structural_path)
        _record("structural-context", structural_path)

    discovery_tiles = _write_discovery_tile_images(frame_dir, inputs.pipeline_input, config, result)
    region_views = _write_region_view_images(frame_dir, inputs.pipeline_input, config, result)

    return StageDebugImages(images=written, discovery_tiles=discovery_tiles, region_views=region_views)
