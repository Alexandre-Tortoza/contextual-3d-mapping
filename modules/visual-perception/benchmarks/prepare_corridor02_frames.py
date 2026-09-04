"""Extrai um conjunto representativo de frames RGB de datasets/raw/corridor-02/corridor-02.bag.

Isto **não** é uma implementação de contract de dataset-adapter (essa fronteira
pertence a `[adapters]`, issues #103/#104, e ainda não foi construída). O
próprio README de `visual-perception` lista explicitamente "dataset/ROS bag
sampling" como uma não-responsabilidade deste módulo. Este script é um gerador
de fixture local e ad-hoc, usado apenas pelos harnesses de benchmark (#174) e
validação end-to-end (#190) deste módulo, que precisam de um conjunto de
imagens reproduzível, real e representativo, e de outra forma não têm como
obtê-lo.

Uso (a partir de `modules/visual-perception`, com o extra `bench` instalado):

    python benchmarks/prepare_corridor02_frames.py

Escreve frames PNG, amostrados uniformemente ao longo da sequência, em
`benchmarks/.local/corridor-02-frames/` (no gitignore — não versionado).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader

DEFAULT_BAG = Path(__file__).resolve().parents[3] / "datasets" / "raw" / "corridor-02" / "corridor-02.bag"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / ".local" / "corridor-02-frames"

_IMAGE_MSGTYPES = {"sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"}
_EXCLUDE_HINTS = ("depth", "infra", "ir")


# Escolhe qual tópico do bag contém as imagens RGB principais: se um tópico
# foi pedido explicitamente, valida e usa esse; senão, filtra tópicos de
# imagem que pareçam ser profundidade/infravermelho e escolhe o com mais
# mensagens (o stream RGB primário, não um secundário raro). Chamada por
# extract_frames para resolver o tópico antes de iterar as mensagens do bag.
def _pick_rgb_topic(reader: AnyReader, requested: str | None) -> str:
    image_connections = [c for c in reader.connections if c.msgtype in _IMAGE_MSGTYPES]
    if not image_connections:
        raise SystemExit(f"No image topics found in bag. All topics: {[c.topic for c in reader.connections]}")

    print("Image topics found in bag:")
    for c in image_connections:
        print(f"  {c.topic!r} ({c.msgtype}, {reader.topics[c.topic].msgcount} messages)")

    if requested is not None:
        if requested not in {c.topic for c in image_connections}:
            raise SystemExit(f"Requested topic {requested!r} is not an image topic in this bag.")
        return requested

    candidates = [c for c in image_connections if not any(h in c.topic.lower() for h in _EXCLUDE_HINTS)]
    candidates = candidates or image_connections
    # Prefer the topic with the most messages: the primary RGB stream, not a rare/aux one.
    best = max(candidates, key=lambda c: reader.topics[c.topic].msgcount)
    return best.topic


# Converte uma mensagem sensor_msgs (Image ou CompressedImage) em um array
# numpy RGB, tratando os encodings bgr8/rgb8/mono8 e o caso comprimido.
# Chamada por extract_frames para cada mensagem selecionada do bag.
def _decode_image(msg: object, msgtype: str) -> np.ndarray:
    """Decodifica uma Image sensor_msgs crua ou comprimida em um array RGB uint8 (H, W, 3).

    Mantida local/manual em vez de usar ``rosbags.image.message_to_cvimage``:
    nos frames ``bgr8`` deste bag esse helper produzia uma coloração errada
    (com tom magenta); um reorder de canais simples com numpy foi verificado
    como correto contra os bytes crus.
    """
    if msgtype == "sensor_msgs/msg/CompressedImage":
        import io

        return np.array(Image.open(io.BytesIO(bytes(msg.data))).convert("RGB"))  # type: ignore[attr-defined]

    height, width, encoding = msg.height, msg.width, msg.encoding  # type: ignore[attr-defined]
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)  # type: ignore[attr-defined]
    if encoding == "bgr8":
        return data.reshape(height, width, 3)[:, :, ::-1]
    if encoding == "rgb8":
        return data.reshape(height, width, 3)
    if encoding == "mono8":
        return np.repeat(data.reshape(height, width, 1), 3, axis=2)
    raise ValueError(f"Unsupported image encoding: {encoding!r}")


# Lê o bag ROS informado, seleciona o tópico RGB (via _pick_rgb_topic), e
# escreve em out_dir uma amostra de `count` frames espaçados uniformemente
# pela sequência, como PNGs numerados. É a função pública usada por main() e
# por outros harnesses (#174, #190) que precisam do conjunto de frames
# corridor-02.
def extract_frames(
    bag_path: Path, out_dir: Path, *, count: int, topic: str | None = None
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with AnyReader([bag_path]) as reader:
        rgb_topic = _pick_rgb_topic(reader, topic)
        connections = [c for c in reader.connections if c.topic == rgb_topic]
        total = reader.topics[rgb_topic].msgcount
        stride = max(total // count, 1)

        written_count = 0
        for index, (connection, _timestamp, rawdata) in enumerate(
            reader.messages(connections=connections)
        ):
            if index % stride == 0 and written_count < count:
                msg = reader.deserialize(rawdata, connection.msgtype)
                image = _decode_image(msg, connection.msgtype)
                out_path = out_dir / f"corridor-02-{written_count:03d}.png"
                Image.fromarray(image).save(out_path)
                written.append(out_path)
                written_count += 1

    return written


# Ponto de entrada de CLI: parseia os argumentos (bag, out_dir, count, topic),
# valida que o bag existe, e chama extract_frames, reportando quantos frames
# foram escritos. Executado via
# `python benchmarks/prepare_corridor02_frames.py`.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--count", type=int, default=18)
    parser.add_argument("--topic", type=str, default=None, help="Override auto-detected RGB topic.")
    args = parser.parse_args()

    if not args.bag.exists():
        raise SystemExit(f"Bag not found: {args.bag}")

    written = extract_frames(args.bag, args.out_dir, count=args.count, topic=args.topic)
    print(f"Wrote {len(written)} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
