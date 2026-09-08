"""Materializa, sob demanda, a evidência que o reasoner recebeu de uma região.

Responde a pergunta que um overlay não responde: quando uma região sai rotulada
como algo que claramente não é, o que exatamente o VLM viu daquela região?

As views (foreground mask-aware, tight crop, contextual crop) não são salvas
pelo run porque são **função pura** de coisas que o run já salva — a observação,
os pixels e a config. Persistir as três para cada uma de dezenas de regiões em
cada frame produziria milhares de arquivos num diretório versionado, para que a
grande maioria nunca fosse aberta. Reconstruí-las aqui custa nada, não usa GPU,
não re-executa o SAM (portanto não sofre com o jitter dele) e produz exatamente
os mesmos pixels, porque é a mesma ``build_region_views`` sobre as mesmas
entradas.

Uso (a partir de ``modules/visual-perception``):

    python benchmarks/inspect_region.py \
      --frame-dir benchmarks/results/samples/<run-id>/frames/corridor-02-002 \
      --region-id region-008b99d686756a04
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for relative in ("src", "../../contracts", "../../adapters/datasets", "../../datasets"):
    sys.path.insert(0, str((_MODULE_ROOT / relative).resolve()))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from render_layers import DrawableShape, binary_mask_image, draw_boxes  # noqa: E402
from visual_perception.application.region_views import build_region_views  # noqa: E402
from visual_perception.config import ModuleConfig  # noqa: E402
from visual_perception.domain.image_payload import ImagePayload  # noqa: E402
from visual_perception.domain.region_evidence import EvidenceSlot  # noqa: E402
from visual_perception.domain.region_reasoning import RegionView  # noqa: E402
from visual_perception.domain.regions import ObservedRegion  # noqa: E402
from visual_perception.domain.visual_observation import VisualObservation  # noqa: E402
from visual_perception.infrastructure.serialization import deserialize_observation  # noqa: E402

#: Nome de arquivo de cada slot de view. ``scene_conditioned`` não aparece: é a
#: imagem inteira, já disponível como ``raw.png`` do frame, e salvá-la uma vez
#: por região seria só duplicação.
_VIEW_FILENAMES = {
    EvidenceSlot.FOREGROUND_DENSE: "foreground.png",
    EvidenceSlot.TIGHT_CROP: "tight.png",
    EvidenceSlot.CONTEXTUAL_CROP: "contextual.png",
}


# Agrupa o que é preciso reconstruir para materializar as views de um frame.
# Existe para que a leitura do disco aconteça em um lugar só, e o resto do
# script trabalhe com objetos de domínio já validados.
@dataclass(frozen=True)
class FrameSnapshot:
    """Uma execução persistida, recarregada a partir dos artifacts do frame.

    Argumentos:
        observation: a observação canônica daquele frame.
        payload: os pixels que o pipeline recebeu.
        config: a configuração exata com que o run foi executado.
    """

    observation: VisualObservation
    payload: ImagePayload
    config: ModuleConfig


# Recarrega um frame persistido. A config vem do manifest do run, e não de um
# default, porque as views dependem dela: reconstruir com outra config
# produziria imagens que não são as que o reasoner recebeu — exatamente o tipo
# de artifact que mente.
def load_frame(frame_dir: Path) -> FrameSnapshot:
    """Recarrega observação, pixels e config a partir do diretório de um frame.

    Argumentos:
        frame_dir: diretório ``frames/<frame-id>`` de um run.
    Retorna:
        o snapshot pronto para reconstruir as views.
    Levanta:
        SystemExit: quando um artifact obrigatório do frame está ausente.
    """
    observation_path = frame_dir / "observation.json"
    manifest_path = frame_dir.parent.parent / "manifest.json"
    if not observation_path.is_file():
        raise SystemExit(f"Missing {observation_path}.")
    if not manifest_path.is_file():
        raise SystemExit(f"Missing {manifest_path}; the run manifest carries the config.")

    # A entrada do pipeline só é persistida quando difere dos pixels de origem.
    # Preferi-la quando existe é o que mantém as views idênticas às do run,
    # inclusive num run que tenha mascarado o ego-veículo.
    pixels_path = frame_dir / "pipeline-input.png"
    if not pixels_path.is_file():
        pixels_path = frame_dir / "raw.png"
    if not pixels_path.is_file():
        raise SystemExit(f"Missing pixels for {frame_dir}.")

    pixels = np.array(Image.open(pixels_path).convert("RGB"))
    manifest = json.loads(manifest_path.read_text())
    return FrameSnapshot(
        observation=deserialize_observation(json.loads(observation_path.read_text())),
        payload=ImagePayload(pixels, width=pixels.shape[1], height=pixels.shape[0]),
        config=ModuleConfig.from_dict(manifest["config"]),
    )


# Resume a semântica de uma região a partir dos contracts, sem reparsear a
# resposta do modelo. As hipóteses alternativas não são um campo: são claims de
# label irmãos, e é assim que elas aparecem aqui.
def region_semantics(
    region: ObservedRegion, views: tuple[RegionView, ...], config: ModuleConfig
) -> dict[str, object]:
    """Retorna o resumo serializável da semântica e da evidência de uma região.

    Argumentos:
        region: a região inspecionada.
        views: as views construídas para ela.
        config: a config do run, que decide quais dessas views o reasoner viu.
    Retorna:
        o resumo serializável da região.
    """
    labels = [claim for claim in region.claims if claim.kind.value == "label"]
    present = {view.slot for view in views}
    # Construída não é o mesmo que entregue: multi_context decide o que é
    # extraído, region_views decide o que chega ao reasoner. Listar só as
    # construídas sugeriria que o modelo viu imagens que nunca recebeu — a
    # mesma classe de erro do overlay que mostrava confiança geométrica como
    # se fosse semântica.
    delivered = {slot for slot in present if slot.value in config.multimodal_reasoning.region_views}
    return {
        "region_id": region.region_id,
        "geometric_confidence": region.geometric_confidence,
        "contributing_proposal_ids": list(region.contributing_proposal_ids),
        "box": {
            "x_min": region.box.x_min,
            "y_min": region.box.y_min,
            "x_max": region.box.x_max,
            "y_max": region.box.y_max,
        },
        "mask_area": region.mask.area(),
        "primary": None
        if not labels
        else {
            "label": labels[0].value,
            # Ausência de score é preservada como ausência: o contract permite
            # ``None``, e trocá-la por um número inventaria uma certeza que o
            # produtor não afirmou.
            "confidence": None if labels[0].confidence is None else labels[0].confidence.value,
            "provenance": labels[0].provenance.producer,
        },
        "alternatives": [
            {
                "label": claim.value,
                "confidence": None if claim.confidence is None else claim.confidence.value,
            }
            for claim in labels[1:]
        ],
        "claims": [
            {
                "kind": claim.kind.value,
                "value": claim.value,
                "confidence": None if claim.confidence is None else claim.confidence.value,
                "raw_response_json": next(
                    (item.raw_response_json for item in claim.evidence if item.raw_response_json),
                    None,
                ),
            }
            for claim in region.claims
        ],
        "views_built": sorted(slot.value for slot in present),
        "views_delivered_to_reasoner": sorted(slot.value for slot in delivered),
        "views_present": sorted(slot.value for slot in present),
        "views_missing": sorted(
            slot.value for slot in _VIEW_FILENAMES if slot not in present
        ),
        "evidence_slots": [
            {"slot": slot.slot.value, "state": slot.state.value, "reason": slot.reason}
            for slot in region.evidence
        ],
    }


# Escreve as views e o resumo semântico de uma região. Chamada pelo CLI depois
# de reconstruir o frame; separada dele para ser testável sem argumentos de
# linha de comando.
def write_region_inspection(
    snapshot: FrameSnapshot, region_id: str, out_dir: Path
) -> dict[str, str]:
    """Materializa mask, box, views e semântica de uma região em ``out_dir``.

    Argumentos:
        snapshot: o frame recarregado do disco.
        region_id: a região a inspecionar.
        out_dir: diretório de saída; criado se não existir.
    Retorna:
        mapa de nome lógico para caminho relativo dos arquivos escritos.
    Levanta:
        SystemExit: quando a região não existe na observação.
    """
    regions = {region.region_id: region for region in snapshot.observation.regions}
    if region_id not in regions:
        raise SystemExit(
            f"Unknown region {region_id!r}. Known ids: {sorted(regions)[:5]}… "
            f"({len(regions)} total)."
        )
    region = regions[region_id]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    binary_mask_image(region.mask.data).save(out_dir / "mask.png")
    written["mask"] = "mask.png"

    base = Image.fromarray(snapshot.payload.pixels.astype(np.uint8), mode="RGB")
    shape = DrawableShape(shape_id=region.region_id, mask=region.mask, box=region.box)
    draw_boxes(base, (shape,), width=2).save(out_dir / "box.png")
    written["box"] = "box.png"

    # A mesma função que o pipeline usou: a correspondência é estrutural, e não
    # uma convenção que possa divergir de uma reimplementação local do crop.
    views = build_region_views(
        snapshot.observation.regions, snapshot.payload, snapshot.config
    ).get(region_id, ())
    for view in views:
        filename = _VIEW_FILENAMES.get(view.slot)
        if filename is None:
            continue
        Image.fromarray(view.payload.pixels.astype(np.uint8), mode="RGB").save(out_dir / filename)
        written[view.slot.value] = filename

    semantics_path = out_dir / "semantics.json"
    semantics_path.write_text(
        json.dumps(region_semantics(region, views, snapshot.config), indent=2)
    )
    written["semantics"] = "semantics.json"
    return written


# Constrói o parser do CLI em um helper testável, no mesmo padrão do validator.
def _argument_parser() -> argparse.ArgumentParser:
    """Retorna o parser da ferramenta de inspeção de região."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-dir", type=Path, required=True)
    parser.add_argument("--region-id", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="default: <frame-dir>/by-region/<region-id>, fora do controle de versão",
    )
    return parser


# Ponto de entrada: recarrega o frame e materializa a região pedida.
def main(argv: list[str] | None = None) -> None:
    """Executa o CLI de inspeção de região."""
    arguments = _argument_parser().parse_args(argv)
    snapshot = load_frame(arguments.frame_dir)
    out_dir = arguments.out_dir or (arguments.frame_dir / "by-region" / arguments.region_id)
    written = write_region_inspection(snapshot, arguments.region_id, out_dir)
    print(f"Wrote {len(written)} artifacts to {out_dir}")
    for name, relative in sorted(written.items()):
        print(f"  {name}: {relative}")


if __name__ == "__main__":
    main()
