"""Testes da resolução de janelas temporais entre os dois relógios da rosbag."""

from __future__ import annotations

import pytest
from mapping_runtime.bag_window import (
    BagWindow,
    Keyframe,
    keyframe_targets_ns,
    play_offset_seconds,
)

SECOND_NS = 1_000_000_000


# A amostragem de keyframes define quanto de GPU o trecho vai custar, então o
# espaçamento pedido precisa produzir exatamente a quantidade esperada.
def test_keyframe_targets_cobrem_a_janela_no_espacamento_pedido() -> None:
    """Confere quantidade, início e fim dos instantes alvo."""
    targets = keyframe_targets_ns(1_000 * SECOND_NS, 30 * SECOND_NS, 2 * SECOND_NS)
    assert len(targets) == 16
    assert targets[0] == 1_000 * SECOND_NS
    assert targets[-1] == 1_030 * SECOND_NS
    assert all(later - earlier == 2 * SECOND_NS for earlier, later in zip(targets, targets[1:], strict=False))


# Intervalo zero é a composição de um keyframe só, que continua sendo o caso
# usado quando o trecho serve apenas para validar geometria.
def test_keyframe_targets_com_intervalo_zero_seleciona_apenas_o_inicio() -> None:
    """Garante o comportamento de janela com um único keyframe."""
    assert keyframe_targets_ns(5 * SECOND_NS, 30 * SECOND_NS, 0) == (5 * SECOND_NS,)


# Duração e intervalo negativos indicam composição inválida e devem falhar
# antes de qualquer leitura de bag.
def test_keyframe_targets_rejeita_janela_invalida() -> None:
    """Valida os limites da janela pedida."""
    with pytest.raises(ValueError, match="duration_ns"):
        keyframe_targets_ns(0, -1, SECOND_NS)
    with pytest.raises(ValueError, match="interval_ns"):
        keyframe_targets_ns(0, SECOND_NS, -1)


# O prefixo existe para o estimator inercial convergir antes do trecho medido;
# ele nunca pode empurrar a reprodução para antes do início do arquivo.
def test_play_offset_recua_o_prefixo_sem_ficar_negativo() -> None:
    """Confere o offset de reprodução com e sem espaço para o prefixo."""
    bag_start = 100 * SECOND_NS
    assert play_offset_seconds(280 * SECOND_NS, bag_start, 3.0) == pytest.approx(177.0)
    assert play_offset_seconds(101 * SECOND_NS, bag_start, 3.0) == 0.0
    with pytest.raises(ValueError, match="lead_s"):
        play_offset_seconds(280 * SECOND_NS, bag_start, -1.0)


# O payload é a fronteira consumida por Makefile, script do FAST-LIO e pela
# composição contextual; ele precisa preservar as três identidades do keyframe.
def test_payload_preserva_as_tres_identidades_do_keyframe() -> None:
    """Confere a serialização estável da janela resolvida."""
    window = BagWindow(
        camera_topic="/camera_1/image_raw",
        start_header_ns=7,
        end_header_ns=37,
        play_offset_s=1.5,
        play_duration_s=33.0,
        lead_s=3.0,
        keyframes=(Keyframe(sequence_index=4234, header_timestamp_ns=7, bag_timestamp_ns=99),),
    )
    payload = window.to_payload()
    assert payload["keyframes"] == [
        {"sequence_index": 4234, "header_timestamp_ns": 7, "bag_timestamp_ns": 99}
    ]
    assert payload["play_offset_s"] == 1.5
    assert payload["camera_topic"] == "/camera_1/image_raw"
