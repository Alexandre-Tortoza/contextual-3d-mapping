"""Deriva, uma única vez, a geometria de área de uma sequência.

Issue: #202. Duas áreas limitam a evidência utilizável de ``corridor-02``: o
círculo útil da lente fisheye e a silhueta do rig. Nenhuma das duas é derivável
dos intrínsecos: o modelo MEI da sequência (ξ = 1,563) tem limite de FOV em
θ ≈ 130° e limite de desprojeção em ρ ≤ 0,83 (≈ 565 px), ambos maiores que a
diagonal da imagem — a vinheta preta é óptica, não geométrica.

A alternativa determinística é esta: medir o círculo **uma vez**, offline, sobre
a sequência inteira, e versionar o resultado como artifact. A pipeline nunca
infere área por limiar de cor em runtime; ela lê a geometria declarada.

Uso::

    python benchmarks/derive_sequence_masks.py --sequence-id corridor-02

O script reescreve ``benchmarks/sequence-masks/<sequence-id>.json`` e imprime o
erro de ajuste, para que a qualidade do ajuste seja auditável junto do número.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent

#: Onde ficam os frames locais da sequência, e onde o artifact é gravado.
FRAMES_DIR = _THIS_DIR / ".local" / "corridor-02-frames"
MASKS_DIR = _THIS_DIR / "sequence-masks"

#: Luminância média abaixo da qual um pixel é considerado fora da lente. É um
#: limiar de *derivação offline*, não de runtime: ele decide o conteúdo do
#: artifact versionado, que depois é revisado por um humano num diff.
_VIGNETTE_LUMINANCE = 35.0

#: Luminância média acima da qual um pixel é certamente cena. A faixa entre os
#: dois limiares é ambígua e fica fora do ajuste, para que sombra profunda não
#: puxe o círculo para dentro.
_SCENE_LUMINANCE = 60.0


# Agrupa o resultado do ajuste com o erro que o qualifica. Existe para que o
# número derivado nunca viaje sem a medida de quão bem ele descreve os pixels.
@dataclass(frozen=True)
class CircleFit:
    """Um círculo ajustado e os pixels que ele classifica errado."""

    center_x: float
    center_y: float
    radius: float
    dark_inside: int
    lit_outside: int


# Ajusta o círculo útil da lente por busca exaustiva sobre centro e raio.
# Exaustiva de propósito: o espaço é pequeno, o resultado é determinístico e
# não depende de inicialização, o que importa mais aqui que a velocidade.
def fit_valid_circle(mean_luminance: np.ndarray) -> CircleFit:
    """Ajusta o círculo que melhor separa vinheta de cena na média da sequência.

    Argumentos:
        mean_luminance: luminância média por pixel ao longo da sequência.
    Retorna:
        o círculo ajustado e a contagem de pixels mal classificados.
    """
    height, width = mean_luminance.shape
    outside = mean_luminance <= _VIGNETTE_LUMINANCE
    inside = mean_luminance > _SCENE_LUMINANCE
    rows, columns = np.mgrid[0:height, 0:width]
    best: CircleFit | None = None
    for center_x in np.arange(width / 2 - 20, width / 2 + 20, 2.0):
        for center_y in np.arange(height / 2 - 20, height / 2 + 30, 2.0):
            distance = np.hypot(columns - center_x, rows - center_y)
            for radius in np.arange(290.0, 340.0, 2.0):
                disc = distance <= radius
                dark_inside = int((disc & outside).sum())
                lit_outside = int((~disc & inside).sum())
                if best is None or dark_inside + lit_outside < best.dark_inside + best.lit_outside:
                    best = CircleFit(
                        float(center_x), float(center_y), float(radius), dark_inside, lit_outside
                    )
    assert best is not None
    return best


# Carrega a sequência e devolve a luminância média por pixel mais os digests
# das entradas. A média entra no ajuste; os digests entram na proveniência, para
# que o artifact diga exatamente de que pixels ele saiu.
def load_sequence(frames_dir: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    """Carrega os frames e devolve a luminância média e os digests de entrada."""
    paths = sorted(frames_dir.glob("*.png"))
    if not paths:
        raise SystemExit(f"No frames found in {frames_dir}.")
    frames = []
    digests = []
    for path in paths:
        frames.append(np.asarray(Image.open(path).convert("L"), dtype=np.float32))
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    return np.stack(frames).mean(axis=0), tuple(digests)


# Escreve o artifact versionado com geometria e proveniência. A silhueta do rig
# já existente é preservada: ela é desenhada por um humano sobre o frame, e um
# reajuste do círculo não deve apagá-la.
def write_artifact(
    path: Path, *, sequence_id: str, width: int, height: int, fit: CircleFit, digests: tuple[str, ...]
) -> None:
    """Grava a geometria derivada, preservando a silhueta do rig já declarada."""
    existing = json.loads(path.read_text()) if path.is_file() else {}
    payload = {
        "sequence_id": sequence_id,
        "image_width": width,
        "image_height": height,
        "valid_area": {
            "circle": {"cx": fit.center_x, "cy": fit.center_y, "r": fit.radius},
        },
        "ego_vehicle": existing.get("ego_vehicle", {"polygons": []}),
        "provenance": {
            "method": (
                "least-misclassified circle over the per-pixel mean luminance of the whole "
                f"sequence; vignette below {_VIGNETTE_LUMINANCE}, scene above {_SCENE_LUMINANCE}"
            ),
            "derived_by": "benchmarks/derive_sequence_masks.py",
            "source_frame_count": len(digests),
            "source_frames_sha256": list(digests),
            "fit_dark_pixels_inside": fit.dark_inside,
            "fit_lit_pixels_outside": fit.lit_outside,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# Ponto de entrada de linha de comando.
def main() -> None:
    """Deriva e grava a geometria de área da sequência pedida."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", default="corridor-02")
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--masks-dir", type=Path, default=MASKS_DIR)
    arguments = parser.parse_args()

    mean_luminance, digests = load_sequence(arguments.frames_dir)
    height, width = mean_luminance.shape
    fit = fit_valid_circle(mean_luminance)
    destination = arguments.masks_dir / f"{arguments.sequence_id}.json"
    write_artifact(
        destination,
        sequence_id=arguments.sequence_id,
        width=width,
        height=height,
        fit=fit,
        digests=digests,
    )
    total = width * height
    print(
        f"{arguments.sequence_id}: circle=({fit.center_x:.1f}, {fit.center_y:.1f}) r={fit.radius:.1f} "
        f"dark-inside={fit.dark_inside} lit-outside={fit.lit_outside} "
        f"({(fit.dark_inside + fit.lit_outside) / total:.2%} of the frame) -> {destination}"
    )


if __name__ == "__main__":
    main()
