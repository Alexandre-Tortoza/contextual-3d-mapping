"""Resolução de janelas temporais e keyframes entre os dois relógios de uma rosbag."""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Descreve um keyframe RGB pelas três identidades que os consumidores precisam:
# a posição no stream, o instante do header (relógio do dataset, usado pela
# associação e pelo ground-truth) e o instante de gravação (relógio do bag,
# usado por rosbag play). Existe porque nenhuma delas é derivável das outras.
@dataclass(frozen=True)
class Keyframe:
    """Identidade completa de um frame RGB selecionado.

    Argumentos:
        sequence_index: posição do frame no stream RGB do bag.
        header_timestamp_ns: timestamp do header ROS, no relógio do dataset.
        bag_timestamp_ns: timestamp de gravação, no relógio do arquivo.
    """

    sequence_index: int
    header_timestamp_ns: int
    bag_timestamp_ns: int


# Agrupa tudo que um trecho precisa publicar: como pedir esse trecho ao
# ``rosbag play`` e quais frames RGB o compõem. É a saída consumida pelo
# Makefile, pelo script do FAST-LIO e pela composição contextual.
@dataclass(frozen=True)
class BagWindow:
    """Janela temporal resolvida de uma rosbag.

    Argumentos:
        camera_topic: tópico RGB usado como referência temporal.
        start_header_ns: início da janela no relógio do dataset.
        end_header_ns: fim da janela no relógio do dataset.
        play_offset_s: valor de ``rosbag play --start`` que alcança a janela.
        play_duration_s: valor de ``rosbag play --duration`` correspondente.
        lead_s: prefixo reproduzido antes da janela para o estimator convergir.
        keyframes: frames RGB selecionados dentro da janela.
    """

    camera_topic: str
    start_header_ns: int
    end_header_ns: int
    play_offset_s: float
    play_duration_s: float
    lead_s: float
    keyframes: tuple[Keyframe, ...]

    # Serializa para o formato consumido por scripts e pela composição, sem
    # expor objetos de terceiros nem o layout interno do bag.
    def to_payload(self) -> dict[str, Any]:
        """Converte a janela em um payload JSON estável."""
        return {
            "camera_topic": self.camera_topic,
            "start_header_ns": self.start_header_ns,
            "end_header_ns": self.end_header_ns,
            "play_offset_s": self.play_offset_s,
            "play_duration_s": self.play_duration_s,
            "lead_s": self.lead_s,
            "keyframes": [
                {
                    "sequence_index": keyframe.sequence_index,
                    "header_timestamp_ns": keyframe.header_timestamp_ns,
                    "bag_timestamp_ns": keyframe.bag_timestamp_ns,
                }
                for keyframe in self.keyframes
            ],
        }


# Calcula os instantes de keyframe pedidos dentro da janela. Fica separado da
# leitura do bag porque a política de amostragem é uma decisão de composição,
# não uma propriedade do arquivo, e assim permanece testável sem uma rosbag.
def keyframe_targets_ns(start_ns: int, duration_ns: int, interval_ns: int) -> tuple[int, ...]:
    """Gera os instantes alvo de keyframe dentro de uma janela.

    Argumentos:
        start_ns: início da janela no relógio do dataset.
        duration_ns: duração da janela em nanossegundos.
        interval_ns: espaçamento pedido entre keyframes; zero seleciona só o primeiro.
    Retorna:
        instantes alvo em ordem crescente, todos dentro da janela.
    Levanta:
        ValueError: se a duração for negativa ou o intervalo for negativo.
    """
    if duration_ns < 0:
        raise ValueError("duration_ns must be non-negative.")
    if interval_ns < 0:
        raise ValueError("interval_ns must be non-negative.")
    if interval_ns == 0:
        return (start_ns,)
    count = duration_ns // interval_ns + 1
    return tuple(start_ns + index * interval_ns for index in range(count))


# Converte um instante de gravação no argumento de ``rosbag play --start``,
# recuando o prefixo pedido. O recuo existe porque o estimator inercial precisa
# de alguns segundos de IMU antes do trecho que será mapeado.
def play_offset_seconds(window_bag_ns: int, bag_start_ns: int, lead_s: float) -> float:
    """Calcula o offset de reprodução que alcança a janela.

    Argumentos:
        window_bag_ns: início da janela no relógio de gravação.
        bag_start_ns: primeiro instante de gravação do bag.
        lead_s: prefixo reproduzido antes da janela.
    Retorna:
        offset em segundos, nunca negativo.
    Levanta:
        ValueError: se o prefixo for negativo.
    """
    if lead_s < 0:
        raise ValueError("lead_s must be non-negative.")
    return max((window_bag_ns - bag_start_ns) / 1_000_000_000 - lead_s, 0.0)


# Extrai o timestamp do header ROS. O bag preserva dois relógios, e só o header
# é comparável com o ground-truth e com a associação sensorial.
def _header_timestamp_ns(message: Any) -> int:
    """Retorna o timestamp inteiro do header de uma mensagem ROS."""
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


# Lê o header de uma única mensagem identificada pelo instante de gravação.
# Existe para que a busca por instante do dataset não precise desserializar o
# stream inteiro, que neste dataset são dezenas de gigabytes de imagem.
def _header_at(reader: Any, connections: list[Any], bag_timestamp_ns: int) -> int:
    """Lê o timestamp de header da mensagem gravada em um instante conhecido."""
    for connection, _, rawdata in reader.messages(
        connections=connections, start=bag_timestamp_ns, stop=bag_timestamp_ns + 1
    ):
        return _header_timestamp_ns(reader.deserialize(rawdata, connection.msgtype))
    raise ValueError(f"no message recorded at bag timestamp {bag_timestamp_ns}.")


# Localiza o frame RGB cujo header está mais próximo do instante pedido. Usa o
# desvio local entre os dois relógios para estimar a posição e depois corrige
# caminhando pelos vizinhos, porque os relógios do dataset derivam entre si
# (dezenas de milissegundos ao longo da sequência) e uma constante global erra.
def _locate(
    reader: Any,
    connections: list[Any],
    bag_times: list[int],
    target_header_ns: int,
    clock_offset_ns: int,
) -> Keyframe:
    """Resolve o keyframe mais próximo de um instante do relógio do dataset."""
    index = min(bisect_left(bag_times, target_header_ns + clock_offset_ns), len(bag_times) - 1)
    best = (abs(_header_at(reader, connections, bag_times[index]) - target_header_ns), index)
    step = 1 if _header_at(reader, connections, bag_times[index]) < target_header_ns else -1
    candidate = index + step
    while 0 <= candidate < len(bag_times):
        distance = abs(_header_at(reader, connections, bag_times[candidate]) - target_header_ns)
        if distance >= best[0]:
            break
        best = (distance, candidate)
        candidate += step
    resolved = best[1]
    return Keyframe(
        sequence_index=resolved,
        header_timestamp_ns=_header_at(reader, connections, bag_times[resolved]),
        bag_timestamp_ns=bag_times[resolved],
    )


# Resolve uma janela do dataset em instruções concretas de reprodução e em
# keyframes identificados. É o ponto de entrada usado pela CLI para que scripts
# e Makefile nunca precisem embutir a diferença entre os dois relógios do bag.
def resolve_bag_window(
    bag: Path,
    *,
    start_s: float,
    duration_s: float,
    keyframe_interval_s: float = 0.0,
    lead_s: float = 3.0,
    camera_topic: str = "/camera_1/image_raw",
) -> BagWindow:
    """Resolve uma janela temporal e seus keyframes RGB em uma rosbag.

    Argumentos:
        bag: rosbag de origem.
        start_s: início da janela, em segundos após o primeiro frame RGB.
        duration_s: duração da janela em segundos.
        keyframe_interval_s: espaçamento entre keyframes; zero seleciona só o primeiro.
        lead_s: prefixo reproduzido antes da janela para o estimator convergir.
        camera_topic: tópico RGB usado como referência temporal.
    Retorna:
        janela resolvida com offset de reprodução e keyframes identificados.
    Levanta:
        ValueError: se o tópico não existir, a janela for negativa ou exceder o bag.
    """
    from rosbags.highlevel import AnyReader

    if start_s < 0 or duration_s < 0:
        raise ValueError("start_s and duration_s must be non-negative.")
    with AnyReader([bag]) as reader:
        connections = [item for item in reader.connections if item.topic == camera_topic]
        if not connections:
            raise ValueError(f"bag does not contain the RGB topic {camera_topic}.")
        identifiers = {item.id for item in connections}
        bag_times = sorted(
            entry.time
            for source in reader.readers
            for connection_id, entries in source.indexes.items()
            if connection_id in identifiers
            for entry in entries
        )
        if not bag_times:
            raise ValueError(f"bag does not contain messages on {camera_topic}.")
        first_header_ns = _header_at(reader, connections, bag_times[0])
        clock_offset_ns = bag_times[0] - first_header_ns
        start_header_ns = first_header_ns + int(start_s * 1_000_000_000)
        duration_ns = int(duration_s * 1_000_000_000)
        targets = keyframe_targets_ns(
            start_header_ns, duration_ns, int(keyframe_interval_s * 1_000_000_000)
        )
        keyframes: list[Keyframe] = []
        for target in targets:
            keyframe = _locate(reader, connections, bag_times, target, clock_offset_ns)
            if keyframes and keyframe.sequence_index == keyframes[-1].sequence_index:
                continue
            keyframes.append(keyframe)
        if keyframes[0].header_timestamp_ns > start_header_ns + duration_ns:
            raise ValueError("requested window starts after the last RGB frame.")
        window_bag_ns = keyframes[0].bag_timestamp_ns
        return BagWindow(
            camera_topic=camera_topic,
            start_header_ns=start_header_ns,
            end_header_ns=start_header_ns + duration_ns,
            play_offset_s=play_offset_seconds(window_bag_ns, reader.start_time, lead_s),
            play_duration_s=duration_s + lead_s,
            lead_s=lead_s,
            keyframes=tuple(keyframes),
        )


# Persiste a janela resolvida ao lado dos demais artifacts do trecho, para que
# a seleção de keyframes seja reproduzível e auditável depois da execução.
def export_bag_window(window: BagWindow, destination: Path) -> Path:
    """Grava a janela resolvida como JSON."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(window.to_payload(), indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination
