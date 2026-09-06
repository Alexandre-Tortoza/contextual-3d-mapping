"""Pacote HTML de revisão do conjunto de referência anotado.

Issue: #197.

A ferramenta de extração (``prepare_reference.py``) produz amostras
``pending_review``. Este módulo transforma esse manifest em uma página HTML
autocontida, para que a revisão humana — decidir o que anotar, o que ignorar
e o que excluir — aconteça olhando os frames, e não lendo JSON.

A página é autocontida de propósito: as imagens vão embutidas como data URI,
então o pacote pode ser aberto em qualquer máquina sem acesso aos dados
brutos, que continuam fora do controle de versão.

Nada aqui inventa anotação. A página mostra o que foi *medido* (condição de
captura, resolução, digest) e o que ainda falta (estado de revisão, contagem
de regiões), junto da política que define como anotar.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
from collections import Counter
from pathlib import Path

from contextual_mapping_datasets import (
    ReferenceManifest,
    SampleAnnotation,
    Split,
    load_reference_manifest,
)
from contextual_mapping_datasets.annotation_manifest import ReviewState

from .paths import REFERENCE_FRAMES, REFERENCE_MANIFEST, RESULTS_ROOT

#: Itens que o revisor precisa confirmar em cada amostra. Vêm da política de
#: anotação e aparecem na página para que a revisão siga a mesma regra em
#: todas as amostras.
REVIEW_CHECKLIST = (
    "As regiões relevantes para mapeamento estão anotadas, incluindo estruturas pequenas e finas.",
    "Regiões degradadas, ocluídas ou cortadas pela borda estão marcadas como ignoradas.",
    "Labels ambíguos listam todas as alternativas aceitáveis, em vez de escolher uma.",
    "Regiões sem interpretação possível estão marcadas como 'unknown', não rotuladas por suposição.",
    "As relações anotadas referenciam apenas regiões anotadas nesta mesma amostra.",
)


# Constrói o pacote HTML de revisão a partir de um manifest carregado.
# Separada de main para que os testes verifiquem o conteúdo gerado sem
# escrever em disco na árvore do repositório.
def render_review_page(
    manifest: ReferenceManifest, *, frames_dir: Path, embed_images: bool = True
) -> str:
    """Renderiza a página de revisão de ``manifest`` como HTML autocontido.

    Argumentos:
        manifest: o conjunto de referência a revisar.
        frames_dir: onde os frames extraídos estão gravados.
        embed_images: embute as imagens como data URI; desligar produz uma
            página de texto, útil quando os frames não estão disponíveis.
    Retorna:
        o documento HTML completo.
    """
    review_counts = Counter(sample.review_state.value for sample in manifest.samples)
    parts = [
        "<!doctype html>",
        '<html lang="pt-BR"><head><meta charset="utf-8">',
        f"<title>Revisão — {html.escape(manifest.reference_id)}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>Revisão do conjunto de referência <code>{html.escape(manifest.reference_id)}</code></h1>",
        f"<p>Política de anotação: <code>{html.escape(manifest.policy_uri)}</code></p>",
        _summary(manifest, review_counts),
        _checklist(),
    ]
    for split in Split:
        samples = manifest.samples_in(split)
        if not samples:
            continue
        parts.append(f"<h2>Split <code>{split.value}</code> — {len(samples)} amostras</h2>")
        for sample in samples:
            parts.append(_sample_card(sample, frames_dir, embed_images))
    parts.append("</body></html>")
    return "\n".join(parts)


# Grava o pacote de revisão em disco e devolve o caminho do arquivo.
def write_review_package(
    manifest: ReferenceManifest,
    *,
    frames_dir: Path = REFERENCE_FRAMES,
    output_dir: Path | None = None,
    embed_images: bool = True,
) -> Path:
    """Grava a página de revisão e retorna o caminho do arquivo gerado."""
    destination = (output_dir or RESULTS_ROOT / "review" / manifest.reference_id) / "index.html"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_review_page(manifest, frames_dir=frames_dir, embed_images=embed_images),
        encoding="utf-8",
    )
    return destination


# Renderiza o resumo do estado de revisão do conjunto inteiro. Existe para
# que o revisor veja de imediato quanto falta, sem contar cartões.
def _summary(manifest: ReferenceManifest, review_counts: Counter[str]) -> str:
    """Renderiza o bloco de resumo do conjunto."""
    annotated = sum(1 for sample in manifest.samples if sample.regions)
    rows = [
        f"<li>amostras: <strong>{len(manifest.samples)}</strong></li>",
        f"<li>com ao menos uma região anotada: <strong>{annotated}</strong></li>",
    ]
    for state in ReviewState:
        rows.append(f"<li>{state.value}: <strong>{review_counts.get(state.value, 0)}</strong></li>")
    for split in Split:
        rows.append(f"<li>split {split.value}: <strong>{len(manifest.samples_in(split))}</strong></li>")
    return f'<section class="summary"><h2>Resumo</h2><ul>{"".join(rows)}</ul></section>'


# Renderiza a checklist da política de anotação.
def _checklist() -> str:
    """Renderiza a checklist que o revisor aplica a cada amostra."""
    items = "".join(f"<li>{html.escape(item)}</li>" for item in REVIEW_CHECKLIST)
    return f'<section class="checklist"><h2>Checklist de revisão</h2><ol>{items}</ol></section>'


# Renderiza o cartão de uma amostra: imagem, identidade, condições medidas e
# o que já foi anotado.
def _sample_card(sample: SampleAnnotation, frames_dir: Path, embed_images: bool) -> str:
    """Renderiza o cartão de revisão de uma amostra."""
    conditions = ", ".join(sample.capture_conditions) or "—"
    region_lines = "".join(
        f"<li><code>{html.escape(region.annotation_id)}</code>: "
        f"{html.escape(', '.join(region.labels) or '(sem label)')} "
        f"[{region.certainty.value}{', ignorada' if region.ignored else ''}]</li>"
        for region in sample.regions
    )
    regions = f"<ul>{region_lines}</ul>" if region_lines else " nenhuma ainda."
    return (
        f'<article class="sample">'
        f"<h3><code>{html.escape(sample.sample_id)}</code></h3>"
        f"{_image_tag(frames_dir / f'{sample.sample_id}.png', sample, embed_images)}"
        f"<ul>"
        f"<li>resolução: {sample.width}x{sample.height}</li>"
        f"<li>condições medidas: {html.escape(conditions)}</li>"
        f"<li>estado: <strong>{sample.review_state.value}</strong></li>"
        f"<li>frame de origem: <code>{html.escape(str(sample.source_frame_id))}</code></li>"
        f"<li>digest: <code>{html.escape(sample.artifact.digest[:16])}…</code></li>"
        f"</ul>"
        f'<div class="regions"><strong>Regiões anotadas:</strong>{regions}</div>'
        f"</article>"
    )


# Embute a imagem como data URI, ou explica por que ela não está disponível.
# Existe para que a página funcione mesmo em uma máquina sem os dados brutos.
def _image_tag(path: Path, sample: SampleAnnotation, embed_images: bool) -> str:
    """Renderiza a imagem com masks embutida, ou uma nota quando indisponível."""
    if not embed_images:
        return ""
    if not path.exists():
        return (
            '<p class="missing">Frame não disponível localmente: '
            f"<code>{html.escape(path.name)}</code></p>"
        )
    encoded = base64.b64encode(_overlay_bytes(path, sample)).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(path.stem)}">'


# Desenha as masks e labels do rascunho sobre o frame para que o humano
# revise geometria e semântica juntas, sem depender de ferramenta externa.
def _overlay_bytes(path: Path, sample: SampleAnnotation) -> bytes:
    """Retorna um PNG com overlays das regiões anotadas da amostra."""
    from PIL import Image, ImageDraw

    image = Image.open(path).convert("RGBA")
    colors = ((230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48))
    for index, region in enumerate(sample.regions):
        mask_bytes = _decode_mask_bytes(region.mask.runs, sample.width * sample.height)
        mask = Image.frombytes("L", (sample.width, sample.height), mask_bytes)
        color = colors[index % len(colors)]
        layer = Image.new("RGBA", image.size, (*color, 0))
        layer.putalpha(mask.point(lambda value: 90 if value else 0))
        image = Image.alpha_composite(image, layer)
        box = mask.getbbox()
        if box is not None:
            draw = ImageDraw.Draw(image)
            label = ", ".join(region.labels) or "unknown"
            draw.rectangle(box, outline=(*color, 255), width=2)
            draw.text((box[0] + 2, box[1] + 2), label, fill=(255, 255, 255, 255), stroke_width=2)
    output = io.BytesIO()
    image.convert("RGB").save(output, format="PNG")
    return output.getvalue()


# Decodifica o RLE começando em fundo diretamente para bytes 0/255. Existe
# para manter o pacote de revisão leve e independente de numpy/evaluation.
def _decode_mask_bytes(runs: tuple[int, ...], expected_size: int) -> bytes:
    """Decodifica uma máscara RLE validada para bytes de imagem L."""
    values = bytearray()
    foreground = False
    for run in runs:
        values.extend(bytes((255 if foreground else 0,)) * run)
        foreground = not foreground
    if len(values) != expected_size:
        raise ValueError(
            f"Mask RLE covers {len(values)} pixels, expected {expected_size}."
        )
    return bytes(values)


_STYLE = """
body { font-family: system-ui, sans-serif; margin: 2rem auto; max-width: 60rem; line-height: 1.5; }
h1, h2, h3 { line-height: 1.2; }
.summary, .checklist { background: #f5f5f5; padding: 1rem 1.5rem; border-radius: 6px; }
.sample { border: 1px solid #ddd; border-radius: 6px; padding: 1rem; margin: 1rem 0; }
.sample img { max-width: 100%; border-radius: 4px; display: block; }
.missing { color: #a33; font-style: italic; }
code { background: #eee; padding: 0 .25em; border-radius: 3px; }
"""


# Interface de linha de comando do pacote de revisão.
def main() -> None:
    """Gera o pacote de revisão a partir do manifest versionado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--frames-dir", type=Path, default=REFERENCE_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-images", action="store_true")
    arguments = parser.parse_args()

    manifest = load_reference_manifest(arguments.manifest)
    destination = write_review_package(
        manifest,
        frames_dir=arguments.frames_dir,
        output_dir=arguments.output_dir,
        embed_images=not arguments.no_images,
    )
    print(f"Wrote the review package for {len(manifest.samples)} samples to {destination}")


if __name__ == "__main__":  # pragma: no cover - ponto de entrada de CLI
    main()
