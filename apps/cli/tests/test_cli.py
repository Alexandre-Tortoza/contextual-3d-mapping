"""Testes dos contracts de linha de comando sem GPU ou subprocessos reais."""

from __future__ import annotations

from pathlib import Path

import pytest
from contextual_mapping_cli.main import _validate_all_frames_confirmation
from contextual_mapping_cli.models import ExtractionRequest


# Uma confirmação booleana não é suficiente para um job que pode abranger
# dezenas de milhares de chamadas de modelo.
def test_all_frames_exige_segment_id_exato() -> None:
    """Confere a confirmação forte do modo integral."""
    request = ExtractionRequest(Path("run.bag"), "run-full", all_frames=True)
    with pytest.raises(ValueError, match="confirm-all-frames"):
        _validate_all_frames_confirmation(request, None)
    with pytest.raises(ValueError, match="confirm-all-frames"):
        _validate_all_frames_confirmation(request, "sim")
    _validate_all_frames_confirmation(request, "run-full")


# Amostragens comuns não devem ganhar uma confirmação extra que prejudicaria
# automação e o fluxo rápido de trechos pequenos.
def test_amostragem_nao_exige_confirmacao_integral() -> None:
    """Confere que o guard só se aplica a todos os frames."""
    request = ExtractionRequest(Path("run.bag"), "run-sample")
    _validate_all_frames_confirmation(request, None)
