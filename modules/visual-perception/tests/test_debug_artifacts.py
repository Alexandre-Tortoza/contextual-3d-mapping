"""Teste da API pública de imagens de debug (#Passo 1 do schema_version 3).

Cobre ``write_stage_debug_images`` isoladamente, sem depender do layout de
benchmark: quem quiser gerar essas imagens (o harness ou um consumidor de
produção) só precisa de um resultado do pipeline canônico e da config usada.
"""

from __future__ import annotations

from pathlib import Path

from fixtures import default_config, image_observation, payload_with_blobs
from fixtures_ports import default_ports
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.debug_artifacts import FrameInputs, write_stage_debug_images
from visual_perception.domain.image_payload import ImagePayload


# Garante que a função pública escreve as camadas essenciais e devolve seus
# caminhos relativos ao diretório do frame, sem exigir o layout completo de
# benchmark (observation.json, diagnostics.json etc, que ficam com o harness).
def test_write_stage_debug_images_writes_the_core_layers(tmp_path: Path) -> None:
    """As camadas raw/proposals/regions-* saem escritas e mapeadas em ``images``."""
    payload = payload_with_blobs(blobs=((4, 4, 12, 12, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())
    frame_dir = tmp_path / "frame"

    stage_images = write_stage_debug_images(
        frame_dir,
        inputs=FrameInputs(raw=ImagePayload(payload.pixels.copy(), 32, 32), pipeline_input=payload),
        result=result,
        config=default_config(),
    )

    for name in ("raw", "proposals", "regions-masks", "regions-boxes", "regions-labels", "regions-overlay"):
        assert name in stage_images.images
        assert (frame_dir / stage_images.images[name]).is_file()
    # Sem grounding aceito, o preview semântico não é gerado.
    assert "semantic-overlay" not in stage_images.images


# As passadas de discovery e as views por região são a parte da função que o
# layout de benchmark hoje escondia atrás da opção "full debug". Aqui elas são
# sempre produzidas, porque a decisão de chamar ou não a função é de quem chama.
def test_write_stage_debug_images_writes_discovery_and_region_views(tmp_path: Path) -> None:
    """Discovery tiles e region views aparecem com seus caminhos relativos."""
    payload = payload_with_blobs(blobs=((4, 4, 12, 12, (200, 30, 30)),))
    result = run_canonical_pipeline(image_observation(), payload, default_config(), default_ports())
    frame_dir = tmp_path / "frame"

    stage_images = write_stage_debug_images(
        frame_dir,
        inputs=FrameInputs(raw=ImagePayload(payload.pixels.copy(), 32, 32), pipeline_input=payload),
        result=result,
        config=default_config(),
    )

    assert stage_images.discovery_tiles
    for tile in stage_images.discovery_tiles:
        assert (frame_dir / tile["input"]).is_file()
        assert (frame_dir / tile["proposals"]).is_file()
    assert stage_images.region_views
    for view in stage_images.region_views:
        assert "region_id" in view
