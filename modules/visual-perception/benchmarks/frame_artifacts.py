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
    embeddings.npz         os vetores por região, referenciados por artifact_ref
    raw.png               os pixels de origem, nunca modificados
    proposals.png         masks + boxes antes do merge, sem semântica
    regions-masks.png     só as masks finais
    regions-boxes.png     só as boxes finais
    regions-labels.png    labels no centróide, sem caixas
    regions-overlay.png   tudo junto
    structural-context.png  as superfícies não publicadas, esmaecidas
    semantic-overlay.png  máscaras grounded, só quando alguma região tem uma
    valid-area-mask.png   área útil do sensor, quando declarada
    ego-mask.png          máscara de exclusão, quando existe
    pipeline-input.png    só quando difere de raw.png
    discovery/<tile>-input.png, <tile>-proposals.png  cada passada do discovery
    regions/<region-id>/masked-subject.png, tight-crop.png, contextual-crop.png
    DEBUG/<frame-id>-grounding.json  trilha de grounding, só quando houve grounding
```

A escrita das camadas de imagem (``raw.png`` até ``structural-context.png``,
``discovery/`` e ``regions/``) é delegada a
``visual_perception.debug_artifacts.write_stage_debug_images``: este arquivo
continua dono só do que é específico do layout de benchmark — observation.json,
embeddings.npz, diagnostics.json e a trilha de grounding.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from visual_perception.application.observation_diagnostics import ObservationDiagnostics
from visual_perception.application.pipeline import PipelineResult
from visual_perception.config import ModuleConfig
from visual_perception.debug_artifacts import FrameInputs, write_stage_debug_images
from visual_perception.infrastructure.debug_recorder import DebugRecorder
from visual_perception.infrastructure.embedding_archive import write_embedding_archive
from visual_perception.infrastructure.serialization import serialize_observation
from visual_perception.rendering.layers import binary_mask_image

#: Versão do layout de artifacts por frame, gravada no manifest. Um leitor que
#: espera outro layout falha explicitamente em vez de ler o diretório errado em
#: silêncio. A ``frames/4`` move ``discovery/`` e ``regions/`` de dentro de
#: ``DEBUG/`` para o nível do frame, porque a escrita dessas imagens deixou de
#: ser condicionada a um debug "completo" opcional: ``write_stage_debug_images``
#: (API pública do módulo) sempre as produz, e a decisão de chamá-la passou a
#: ser da camada de chamada, não deste layout. A ``frames/3`` havia
#: acrescentado ``structural-context.png`` e os registros por região da
#: política de publicação contextual.
FRAME_ARTIFACT_LAYOUT_VERSION = "frames/4"


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
    config: ModuleConfig,
) -> dict[str, str]:
    """Persiste as camadas de inspeção de um frame e retorna seus caminhos relativos.

    Argumentos:
        frame_dir: diretório do frame; criado se não existir.
        inputs: pixels de origem, pixels entregues ao pipeline e máscaras.
        result: resultado do pipeline, incluindo as proposals pré-merge.
        diagnostics: o resumo estatístico já calculado para a observação.
        extra_diagnostics: campos adicionais do harness (identidade do frame,
            hash da entrada, latência) mesclados no ``diagnostics.json``.
        config: configuração efetiva da run, repassada a
            ``write_stage_debug_images`` para a malha de tiles do discovery e
            as views por região.
    Retorna:
        mapa de nome lógico do artifact para o caminho relativo a ``frame_dir``.
    """
    frame_dir.mkdir(parents=True, exist_ok=True)
    grounding_diagnostics = [
        dict(region.grounding.diagnostics)
        for region in result.observation.regions
        if region.grounding is not None
    ]
    # A trilha de grounding só existe quando houve grounding: um arquivo com a
    # lista vazia por frame não informa nada e poluía o run versionado.
    if grounding_diagnostics:
        DebugRecorder(frame_dir / "DEBUG").record_grounding(frame_dir.name, grounding_diagnostics)

    written: dict[str, str] = {}

    # Um write_text por artifact, com o nome lógico registrado no mesmo passo,
    # para que o manifest nunca liste um caminho que não foi escrito.
    def _record(name: str, path: Path) -> None:
        written[name] = str(path.relative_to(frame_dir))

    observation_path = frame_dir / "observation.json"
    observation_path.write_text(
        json.dumps(serialize_observation(result.observation), indent=2), encoding="utf-8"
    )
    _record("observation", observation_path)

    # Os vetores viajam **por referência**: a observação guarda o
    # ``artifact_ref`` de cada slot, e o arquivo abaixo é o artifact que aquela
    # referência resolve. Embutí-los no JSON inflaria a observação canônica em
    # duas ordens de grandeza sem tornar nada mais auditável.
    all_embeddings = (*result.visual_embeddings, *result.language_embeddings)
    if all_embeddings:
        embeddings_path = frame_dir / "embeddings.npz"
        write_embedding_archive(embeddings_path, all_embeddings)
        _record("embeddings", embeddings_path)

    # Delega a escrita de todas as camadas de imagem (raw, proposals,
    # regions-*, semantic-overlay, structural-context, discovery/ e regions/)
    # à API pública do módulo. O manifest de benchmark mantém as mesmas chaves
    # de sempre (com sublinhado), traduzidas a partir das chaves com hífen que
    # ``write_stage_debug_images`` usa para casar com os nomes de arquivo.
    stage_images = write_stage_debug_images(frame_dir, inputs=inputs, result=result, config=config)
    for name, path in stage_images.images.items():
        written[name.replace("-", "_")] = path
    # Escrito apenas quando difere: dois PNGs idênticos por frame não informam
    # nada, e a igualdade fica afirmada no diagnóstico, não por comparação
    # manual de arquivos.
    if not inputs.pipeline_input_is_raw():
        pipeline_input_path = frame_dir / "pipeline-input.png"
        Image.fromarray(inputs.pipeline_input.pixels.astype(np.uint8), mode="RGB").save(pipeline_input_path)
        _record("pipeline_input", pipeline_input_path)
    if inputs.valid_area_mask is not None:
        valid_path = frame_dir / "valid-area-mask.png"
        binary_mask_image(inputs.valid_area_mask).save(valid_path)
        _record("valid_area_mask", valid_path)
    if inputs.ego_mask is not None:
        ego_path = frame_dir / "ego-mask.png"
        binary_mask_image(inputs.ego_mask).save(ego_path)
        _record("ego_mask", ego_path)

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
        "semantic_grounding": grounding_diagnostics,
        "semantic_overlay_written": "semantic_overlay" in written,
        # O histograma por motivo vive em ``diagnostics.suppressed_regions``.
        # Esta lista é a rastreabilidade por região que o histograma não dá:
        # qual região saiu do output público, com que conceito, e por quê.
        "suppressed_region_records": [
            {
                "region_id": record.region_id,
                "suppressed_reason": record.reason.value,
                "concept": record.concept,
            }
            for record in result.suppressed_regions
        ],
    }
    diagnostics_path = frame_dir / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _record("diagnostics", diagnostics_path)

    return written
