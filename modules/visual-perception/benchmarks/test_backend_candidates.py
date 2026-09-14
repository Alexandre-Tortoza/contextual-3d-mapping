"""Testes das métricas de seleção de backend (#249)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import backend_benchmark  # noqa: E402
from candidates.feature_extraction import first_component_coherence  # noqa: E402
from candidates.language_embedding import relative_similarity_margin  # noqa: E402
from run_backend_benchmark import positive_frame_count  # noqa: E402


# Regressão da #249: o sinal do componente principal é arbitrário, e a métrica
# media o objeto num backbone e o fundo em outro. Inverter o sinal dos tokens
# inverte o componente e não pode mudar o score.
def test_component_coherence_does_not_depend_on_the_component_sign() -> None:
    """Tokens e tokens negados produzem a mesma coerência."""
    grid = np.zeros((6, 6, 4))
    grid[1:4, 1:4] = (5.0, 1.0, 0.0, 0.0)
    tokens = grid.reshape(36, 4) + np.random.default_rng(3).normal(scale=0.01, size=(36, 4))

    assert first_component_coherence(tokens, 6, 6) == pytest.approx(first_component_coherence(-tokens, 6, 6))
    assert first_component_coherence(tokens, 6, 6) > 0.5


# CLIP e SigLIP produzem similaridades em escalas diferentes; a margem relativa
# precisa ser a mesma sob qualquer escala e viés positivos.
def test_relative_margin_is_invariant_to_similarity_scale_and_bias() -> None:
    """Transformar as similaridades por escala e viés não muda a margem."""
    similarities = np.array([0.21, 0.28, 0.24, 0.19])
    transformed = 10.0 * similarities - 3.5

    assert relative_similarity_margin(similarities) == pytest.approx(relative_similarity_margin(transformed))
    assert 0.0 <= relative_similarity_margin(similarities) <= 1.0
    assert relative_similarity_margin(np.full(4, 0.3)) == 0.0


# Sem CUDA não há VRAM a medir; o benchmark precisa recusar em vez de reportar
# RSS do processo com o nome de VRAM.
def test_benchmark_refuses_to_run_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sem GPU, ``benchmark_candidate`` falha antes de carregar o candidato."""
    monkeypatch.setattr(backend_benchmark, "torch", None)

    def _forbidden_factory() -> object:
        raise AssertionError("the candidate must not load without a way to measure VRAM")

    with pytest.raises(RuntimeError, match="CUDA"):
        backend_benchmark.benchmark_candidate("candidate", _forbidden_factory, lambda model: 1.0)


# ``--frames 0`` virava "todos os frames" e ``-1`` "todos menos um".
@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_frame_count_must_be_a_positive_integer(value: str) -> None:
    """Zero, negativo e texto não numérico são recusados."""
    with pytest.raises(argparse.ArgumentTypeError):
        positive_frame_count(value)
    assert positive_frame_count("3") == 3
