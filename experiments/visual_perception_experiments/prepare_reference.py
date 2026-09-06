"""Extração determinística dos frames do conjunto de referência anotado.

Issue: #197.

Esta ferramenta produz a *estrutura* do conjunto de referência — quais
frames, em que split, com que identidade e proveniência — e deixa as
anotações para revisão humana. Nenhum label é inventado: toda amostra nasce
com ``regions=()`` e ``review_state=pending_review``. Só o que é medível a
partir do próprio sinal (condição de captura fotométrica) é preenchido
automaticamente.

A amostragem é temporal e determinística: a sequência é dividida em três
blocos contíguos — development, calibration, test — separados por *guard
bands* descartadas. As bandas existem porque frames vizinhos de um vídeo são
quase idênticos: sem elas, o mesmo conteúdo apareceria em dois splits e o
resultado de teste estaria contaminado antes da primeira métrica. Dentro de
cada bloco os frames são amostrados uniformemente, então a mesma sequência
sempre produz o mesmo conjunto.

Uso (a partir da raiz do repositório, com o extra ``bench`` instalado):

    python -m visual_perception_experiments.prepare_reference --frames-per-split 12
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from contextual_mapping_datasets import (
    AnnotationProvenance,
    ReferenceManifest,
    SampleAnnotation,
    SampleArtifact,
    Split,
    load_reference_manifest,
    save_reference_manifest,
)
from contextual_mapping_datasets.annotation_manifest import ReviewState

from .frame_conditions import measure_conditions
from .paths import REFERENCE_FRAMES, REFERENCE_MANIFEST, REPOSITORY_ROOT

#: Versão da política de anotação que este conjunto segue. O manifest aponta
#: para o documento; esta constante é o que amarra cada amostra à versão.
ANNOTATION_POLICY_VERSION = "annotation-policy/1"

#: Documento que define o que anotar, o que ignorar e como resolver
#: divergência entre anotadores.
ANNOTATION_POLICY_URI = "modules/visual-perception/docs/annotation-policy.md"

#: Fração da sequência atribuída a cada split, na ordem temporal. Development
#: recebe a maior fatia porque é onde a iteração acontece; test vem por
#: último para que o material menos visto seja o de decisão.
SPLIT_FRACTIONS: tuple[tuple[Split, float], ...] = (
    (Split.DEVELOPMENT, 0.40),
    (Split.CALIBRATION, 0.25),
    (Split.TEST, 0.35),
)

#: Fração da sequência descartada entre dois blocos consecutivos. Frames
#: vizinhos são quase idênticos, então sem essa banda o mesmo conteúdo
#: vazaria de um split para o outro.
GUARD_BAND_FRACTION = 0.03

#: Primeira linha ocupada pelo chassi e pelas rodas do robô que carrega a
#: câmera do corridor-02. A câmera está montada no próprio robô, então essa
#: faixa é a mesma em todo frame e não é conteúdo da cena. Ela é zerada aqui,
#: e não no módulo, porque é conhecimento do rig que gerou os dados (ver
#: AGENTS.md, "Configuration ownership"). Confirmada visualmente em três
#: frames da sequência a 640x480.
EGO_VEHICLE_ROW_START = 340


# Descreve a fatia temporal de um split dentro da sequência, já com as guard
# bands descontadas. Existe para que a divisão seja um valor inspecionável e
# testável, e não um efeito colateral do laço de extração.
@dataclass(frozen=True)
class SplitWindow:
    """A janela de índices de frame atribuída a um split."""

    split: Split
    start: int
    stop: int

    # Valida que a janela é não vazia e bem ordenada, falhando cedo quando a
    # sequência é curta demais para comportar três blocos e as bandas.
    def __post_init__(self) -> None:
        """Rejeita janelas vazias ou invertidas."""
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(
                f"split {self.split.value!r} has an empty frame window: [{self.start}, {self.stop})."
            )

    # Escolhe índices uniformemente espaçados dentro da janela. Existe para
    # que a amostragem seja determinística e cubra a janela inteira, em vez
    # de concentrar em uma ponta.
    def sample(self, count: int) -> tuple[int, ...]:
        """Retorna ``count`` índices uniformemente espaçados dentro da janela."""
        if count <= 0:
            raise ValueError("The number of frames per split must be positive.")
        available = self.stop - self.start
        if count > available:
            raise ValueError(
                f"split {self.split.value!r} has only {available} frames available, "
                f"but {count} were requested."
            )
        step = available / count
        return tuple(self.start + int((index + 0.5) * step) for index in range(count))


# Divide a sequência em blocos temporais separados por guard bands. Existe
# como a única implementação da política de split, para que a ferramenta e
# os testes concordem sobre o que "sem vazamento" significa.
def split_windows(
    total_frames: int,
    *,
    fractions: tuple[tuple[Split, float], ...] = SPLIT_FRACTIONS,
    guard_band_fraction: float = GUARD_BAND_FRACTION,
) -> tuple[SplitWindow, ...]:
    """Divide ``total_frames`` em janelas por split, separadas por guard bands.

    Argumentos:
        total_frames: número de frames disponíveis na sequência.
        fractions: fração da sequência de cada split, em ordem temporal.
        guard_band_fraction: fração descartada entre blocos consecutivos.
    Retorna:
        uma :class:`SplitWindow` por split, na ordem temporal.
    Levanta:
        ValueError: se a sequência for curta demais para a divisão pedida.
    """
    if total_frames <= 0:
        raise ValueError("The sequence must contain at least one frame.")
    guard = int(total_frames * guard_band_fraction)
    windows: list[SplitWindow] = []
    cursor = 0
    for index, (split, fraction) in enumerate(fractions):
        is_last = index == len(fractions) - 1
        stop = total_frames if is_last else cursor + int(total_frames * fraction)
        windows.append(SplitWindow(split, cursor, stop))
        cursor = stop + guard
        if not is_last and cursor >= total_frames:
            raise ValueError(
                f"the sequence has only {total_frames} frames, which is not enough for "
                f"{len(fractions)} split blocks separated by guard bands."
            )
    return tuple(windows)


# Constrói uma amostra do manifest a partir de um frame já gravado em disco.
# Nenhuma anotação é criada: só identidade, proveniência e as condições
# medidas do sinal. Chamada por prepare para cada frame selecionado.
def build_sample(
    sample_id: str,
    frame_path: Path,
    pixels: np.ndarray,
    split: Split,
    source_frame_id: str,
    *,
    dataset_root: Path,
) -> SampleAnnotation:
    """Constrói a amostra ``pending_review`` correspondente a um frame extraído.

    Argumentos:
        sample_id: identidade estável da amostra no conjunto.
        frame_path: onde o frame foi gravado.
        pixels: o array RGB do frame, para medir as condições de captura.
        split: split ao qual a amostra pertence.
        source_frame_id: identidade do frame na sequência de origem.
        dataset_root: raiz do dataset, usada para derivar a uri relativa.
    Retorna:
        a :class:`SampleAnnotation` sem anotações, pronta para revisão.
    """
    return SampleAnnotation(
        sample_id=sample_id,
        artifact=SampleArtifact(
            uri=frame_path.relative_to(dataset_root).as_posix(),
            media_type="image/png",
            digest=hashlib.sha256(frame_path.read_bytes()).hexdigest(),
        ),
        width=int(pixels.shape[1]),
        height=int(pixels.shape[0]),
        split=split,
        provenance=AnnotationProvenance(
            annotator="unassigned",
            method="tool_extracted_pending_annotation",
            policy_version=ANNOTATION_POLICY_VERSION,
            annotated_at=None,
            resolution="not yet reviewed",
        ),
        source_frame_id=source_frame_id,
        capture_conditions=(*measure_conditions(pixels), "ego_vehicle_masked"),
        strata=(),
        regions=(),
        relations=(),
        review_state=ReviewState.PENDING_REVIEW,
    )


# Recusa sobrescrever um manifest que já contém trabalho humano. Existe
# porque reexecutar a ferramenta é normal, e apagar anotações revisadas por
# engano é irreversível.
def guard_existing_manifest(path: Path, *, force: bool) -> None:
    """Levanta se ``path`` já contém anotações ou revisão, a menos que ``force``.

    Argumentos:
        path: destino do manifest.
        force: permite sobrescrever explicitamente.
    Levanta:
        SystemExit: se houver trabalho humano gravado e ``force`` for falso.
    """
    if force or not path.exists():
        return
    existing = load_reference_manifest(path)
    annotated = [
        sample.sample_id
        for sample in existing.samples
        if sample.regions or sample.review_state is not ReviewState.PENDING_REVIEW
    ]
    if annotated:
        raise SystemExit(
            f"{path} already contains reviewed or annotated samples ({len(annotated)} of "
            f"{len(existing.samples)}). Re-running would discard that work; pass --force only if "
            "you intend to replace it."
        )


# Constrói o manifest completo a partir das amostras extraídas. Existe
# separada de prepare para que os testes montem um manifest sem tocar em um
# bag ROS.
def build_manifest(samples: tuple[SampleAnnotation, ...], *, reference_id: str) -> ReferenceManifest:
    """Monta o :class:`ReferenceManifest` das amostras extraídas."""
    return ReferenceManifest(
        reference_id=reference_id,
        dataset_id="corridor-02",
        policy_uri=ANNOTATION_POLICY_URI,
        samples=samples,
        created_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


# Extrai os frames selecionados do bag e grava manifest e imagens. Ponto de
# entrada real da ferramenta; separado de main para ser chamável por outro
# script sem passar por argv.
def prepare(
    bag_path: Path,
    *,
    frames_per_split: int,
    frames_dir: Path = REFERENCE_FRAMES,
    manifest_path: Path = REFERENCE_MANIFEST,
    reference_id: str = "corridor-02-visual-reference-1",
    force: bool = False,
    topic: str | None = None,
) -> ReferenceManifest:
    """Extrai os frames do conjunto de referência e grava o manifest correspondente.

    Argumentos:
        bag_path: bag ROS de origem.
        frames_per_split: quantos frames extrair de cada split.
        frames_dir: destino dos PNGs extraídos (dados brutos, não versionados).
        manifest_path: destino do manifest versionado.
        reference_id: identidade estável do conjunto de referência.
        force: permite sobrescrever um manifest já anotado.
        topic: tópico de imagem a usar; o default escolhe o stream RGB primário.
    Retorna:
        o manifest gravado.
    """
    guard_existing_manifest(manifest_path, force=force)
    frames = _read_frames(bag_path, topic)
    windows = split_windows(len(frames))
    frames_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = frames_dir.parent

    samples: list[SampleAnnotation] = []
    for window in windows:
        for index in window.sample(frames_per_split):
            timestamp, raw_pixels = frames[index]
            pixels = mask_ego_vehicle(raw_pixels)
            sample_id = f"corridor-02-{window.split.value}-{index:06d}"
            frame_path = frames_dir / f"{sample_id}.png"
            _write_png(frame_path, pixels)
            samples.append(
                build_sample(
                    sample_id,
                    frame_path,
                    pixels,
                    window.split,
                    source_frame_id=f"corridor-02:{timestamp}",
                    dataset_root=dataset_root,
                )
            )

    manifest = build_manifest(tuple(samples), reference_id=reference_id)
    save_reference_manifest(manifest, manifest_path)
    return manifest


# Zera a faixa ocupada pelo chassi do robô. Existe porque essa faixa
# aparece em todo frame e não pertence ao ambiente sendo mapeado: mantê-la
# faria o anotador e o modelo gastarem esforço no próprio veículo. O frame
# gravado no conjunto de referência é o mesmo que o pipeline recebe, para
# que anotação e predição enxerguem exatamente os mesmos pixels.
def mask_ego_vehicle(pixels: np.ndarray, *, row_start: int = EGO_VEHICLE_ROW_START) -> np.ndarray:
    """Retorna uma cópia de ``pixels`` com a faixa do chassi zerada."""
    if row_start >= pixels.shape[0]:
        return pixels.copy()
    masked = pixels.copy()
    masked[row_start:, :, :] = 0
    return masked


# Lê o bag e devolve os frames RGB em ordem temporal. Isolado em uma função
# para que a dependência opcional de ``rosbags`` só seja exigida quando a
# extração real acontece.
def _read_frames(bag_path: Path, topic: str | None) -> list[tuple[int, np.ndarray]]:
    """Lê os frames RGB do bag, em ordem crescente de timestamp."""
    import sys

    sys.path.insert(0, str(REPOSITORY_ROOT / "modules" / "visual-perception" / "benchmarks"))
    from rosbags.highlevel import AnyReader

    from prepare_corridor02_frames import _decode_image, _pick_rgb_topic  # type: ignore[import-not-found]

    frames: list[tuple[int, np.ndarray]] = []
    with AnyReader([bag_path]) as reader:
        selected = _pick_rgb_topic(reader, topic)
        connections = [item for item in reader.connections if item.topic == selected]
        for connection, timestamp, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            frames.append((int(timestamp), _decode_image(message, connection.msgtype)))
    frames.sort(key=lambda item: item[0])
    return frames


# Grava um frame como PNG determinístico. Isolado para que a dependência de
# Pillow fique confinada ao caminho de escrita.
def _write_png(path: Path, pixels: np.ndarray) -> None:
    """Grava ``pixels`` como PNG em ``path``."""
    from PIL import Image

    Image.fromarray(pixels).save(path, format="PNG", optimize=False)


# Interface de linha de comando da ferramenta.
def main() -> None:
    """Executa a preparação do conjunto de referência a partir de argumentos de CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        type=Path,
        default=REPOSITORY_ROOT / "datasets" / "raw" / "corridor-02" / "corridor-02.bag",
    )
    parser.add_argument("--frames-per-split", type=int, default=12)
    parser.add_argument("--topic", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()

    manifest = prepare(
        arguments.bag,
        frames_per_split=arguments.frames_per_split,
        force=arguments.force,
        topic=arguments.topic,
    )
    print(f"Wrote {len(manifest.samples)} pending-review samples to {REFERENCE_MANIFEST}")
    for split in Split:
        print(f"  {split.value}: {len(manifest.samples_in(split))} samples")


if __name__ == "__main__":  # pragma: no cover - ponto de entrada de CLI
    main()
