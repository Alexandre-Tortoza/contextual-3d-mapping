"""Conjunto de referência sintético e determinístico para testes locais.

Issue: #197.

Os frames reais do `corridor-02` não são distribuídos: eles ficam em
`datasets/raw/`, fora do controle de versão. Sem um substituto, todo teste
que exercita a cadeia manifest -> execução -> avaliação dependeria de dados
que a maioria das máquinas não tem.

Este módulo constrói um conjunto mínimo equivalente *proceduralmente*: três
amostras, uma por split, com formas geométricas de cor sólida sobre um fundo
neutro, e anotações que descrevem exatamente essas formas. Ele é
determinístico (mesma entrada, mesmos bytes), pequeno, e cobre os casos que
a política de anotação prevê — região comum, região pequena, região ambígua
e região ignorada.

NOTE: as anotações aqui são *verdade por construção*: sabemos onde cada
forma está porque nós a desenhamos. Isso é o oposto de inventar anotação
para dados reais, que a política proíbe.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contextual_mapping_datasets import (
    AnnotationCertainty,
    AnnotationProvenance,
    MaskAnnotation,
    ReferenceManifest,
    RegionAnnotation,
    RelationAnnotation,
    ReviewState,
    SampleAnnotation,
    SampleArtifact,
    Split,
)

#: Resolução das amostras sintéticas. Pequena o suficiente para os testes
#: rodarem rápido, grande o suficiente para o backbone fake produzir uma
#: grade de features com mais de uma célula por região.
FIXTURE_SIZE = 64

#: Identidade do conjunto sintético, distinta da do conjunto real para que
#: um relatório nunca confunda os dois.
FIXTURE_REFERENCE_ID = "synthetic-visual-reference-1"

_PROVENANCE = AnnotationProvenance(
    annotator="fixture",
    method="procedurally_generated",
    policy_version="annotation-policy/1",
    resolution="ground truth by construction",
)


# Descreve uma forma desenhada em uma amostra sintética, junto da anotação
# que ela justifica. Existe para que desenho e anotação venham da mesma
# fonte, e não possam divergir.
@dataclass(frozen=True)
class Shape:
    """Um retângulo desenhado na amostra e a anotação correspondente."""

    annotation_id: str
    x_min: int
    y_min: int
    size: int
    color: tuple[int, int, int]
    labels: tuple[str, ...] = ()
    certainty: AnnotationCertainty = AnnotationCertainty.CERTAIN
    ignored: bool = False

    # Materializa a máscara booleana da forma na resolução da fixture.
    def mask(self, size: int = FIXTURE_SIZE) -> np.ndarray:
        """Retorna a máscara booleana desta forma."""
        mask = np.zeros((size, size), dtype=np.bool_)
        mask[self.y_min : self.y_min + self.size, self.x_min : self.x_min + self.size] = True
        return mask


#: As formas de cada amostra sintética. A amostra de development traz uma
#: região comum e uma pequena; calibration traz uma região ambígua; test
#: traz uma região ignorada ao lado de uma comum.
_SAMPLES: dict[Split, tuple[Shape, ...]] = {
    Split.DEVELOPMENT: (
        Shape("dev-wall", 4, 4, 24, (180, 60, 60), ("parede", "wall")),
        Shape("dev-switch", 44, 10, 4, (60, 180, 60), ("interruptor", "switch")),
    ),
    Split.CALIBRATION: (
        Shape("cal-door", 8, 8, 28, (60, 60, 180), ("porta", "armário"), AnnotationCertainty.AMBIGUOUS),
    ),
    Split.TEST: (
        Shape("test-floor", 4, 34, 26, (200, 160, 40), ("piso", "floor")),
        Shape("test-blurred", 40, 40, 12, (120, 120, 120), (), AnnotationCertainty.UNKNOWN, ignored=True),
    ),
}


# Desenha a imagem de uma amostra sintética a partir das suas formas.
# Chamada por write_fixture_frames e pelos testes que precisam do payload.
def render_frame(shapes: tuple[Shape, ...], *, size: int = FIXTURE_SIZE) -> np.ndarray:
    """Desenha as formas de uma amostra sobre um fundo neutro."""
    pixels = np.full((size, size, 3), 40, dtype=np.uint8)
    for shape in shapes:
        pixels[shape.y_min : shape.y_min + shape.size, shape.x_min : shape.x_min + shape.size] = shape.color
    return pixels


# Codifica uma máscara booleana no RLE do manifest. Existe local ao módulo
# porque a fixture é o único produtor de anotação deste repositório.
def encode_runs(mask: np.ndarray) -> tuple[int, ...]:
    """Codifica ``mask`` em RLE começando por uma contagem ``False``."""
    flat = mask.reshape(-1)
    runs: list[int] = []
    current = False
    count = 0
    for value in flat.tolist():
        if bool(value) == current:
            count += 1
            continue
        runs.append(count)
        current = bool(value)
        count = 1
    runs.append(count)
    return tuple(runs)


# Grava os frames sintéticos em disco e devolve o caminho de cada amostra.
# Existe para que os testes de execução ponta a ponta tenham arquivos reais
# para o runner carregar, sem depender dos dados brutos.
def write_fixture_frames(frames_dir: Path, *, size: int = FIXTURE_SIZE) -> dict[str, Path]:
    """Grava um PNG por amostra sintética e retorna o caminho de cada uma."""
    from PIL import Image

    frames_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split, shapes in _SAMPLES.items():
        sample_id = _sample_id(split)
        path = frames_dir / f"{sample_id}.png"
        Image.fromarray(render_frame(shapes, size=size)).save(path, format="PNG")
        written[sample_id] = path
    return written


# Constrói o manifest sintético, opcionalmente amarrado aos frames já
# gravados para que os digests correspondam aos arquivos reais.
def build_fixture_manifest(frames_dir: Path | None = None, *, size: int = FIXTURE_SIZE) -> ReferenceManifest:
    """Constrói o :class:`ReferenceManifest` sintético, revisado por construção.

    Argumentos:
        frames_dir: quando informado, o digest de cada amostra é o do arquivo
            gravado; caso contrário é derivado do id, o que basta para os
            testes que não carregam pixels.
        size: resolução das amostras.
    Retorna:
        o manifest sintético completo.
    """
    samples: list[SampleAnnotation] = []
    for split, shapes in _SAMPLES.items():
        sample_id = _sample_id(split)
        path = None if frames_dir is None else frames_dir / f"{sample_id}.png"
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path is not None and path.exists()
            else hashlib.sha256(sample_id.encode()).hexdigest()
        )
        regions = tuple(
            RegionAnnotation(
                annotation_id=shape.annotation_id,
                mask=MaskAnnotation(size, size, encode_runs(shape.mask(size))),
                labels=shape.labels,
                certainty=shape.certainty,
                ignored=shape.ignored,
                visibility="degraded" if shape.ignored else "clear",
            )
            for shape in shapes
        )
        relations = (
            (RelationAnnotation("dev-rel-1", "dev-wall", "near", "dev-switch"),)
            if split is Split.DEVELOPMENT
            else ()
        )
        samples.append(
            SampleAnnotation(
                sample_id=sample_id,
                artifact=SampleArtifact(f"visual-reference/{sample_id}.png", "image/png", digest),
                width=size,
                height=size,
                split=split,
                provenance=_PROVENANCE,
                source_frame_id=f"synthetic:{sample_id}",
                capture_conditions=("normal_light", "sharp"),
                strata=("synthetic",),
                regions=regions,
                relations=relations,
                review_state=ReviewState.REVIEWED,
            )
        )
    return ReferenceManifest(
        reference_id=FIXTURE_REFERENCE_ID,
        dataset_id="corridor-02",
        policy_uri="modules/visual-perception/docs/annotation-policy.md",
        samples=tuple(samples),
        created_at="2026-01-01T00:00:00Z",
    )


# Expõe as formas de uma amostra, para que um teste possa comparar uma
# predição com a geometria que ele mesmo desenhou.
def fixture_shapes(split: Split) -> tuple[Shape, ...]:
    """Retorna as formas desenhadas na amostra sintética de ``split``."""
    return _SAMPLES[split]


# Deriva o id estável de uma amostra sintética a partir do seu split.
def _sample_id(split: Split) -> str:
    """Retorna o id da amostra sintética de ``split``."""
    return f"synthetic-{split.value}"
