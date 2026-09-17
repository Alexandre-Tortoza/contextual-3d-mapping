"""Testes da ponte entre ``language_embedding_ref`` e a fusão CLIP (#140).

Cobrem só as fronteiras puras que leem o manifest da run de percepção e
resolvem o archive ``embeddings.npz`` de um frame — não o pipeline completo
de ``export_corridor02_context``, que exige bag, PCD e calibração reais.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from mapping_runtime.corridor02_context import (
    _language_embedding_config,
    _language_embedding_reference,
    _resolve_language_embedding,
)


# Sem manifest, a configuração é ausente por definição: o chamador decide se
# isso é fatal conforme a capability estar ligada.
def test_language_embedding_config_ausente_sem_manifest(tmp_path: Path) -> None:
    """Devolve ``None`` quando não há ``manifest.json`` na run."""
    frame_dir = tmp_path / "run" / "frames" / "frame-000"
    frame_dir.mkdir(parents=True)
    observation = frame_dir / "observation.json"
    observation.write_text("{}", encoding="utf-8")
    assert _language_embedding_config(observation) is None


# Um manifest sem a seção ``language_embedding`` também é ausência explícita,
# não um erro: nem toda run de percepção habilitou o estágio CLIP.
def test_language_embedding_config_ausente_sem_secao(tmp_path: Path) -> None:
    """Devolve ``None`` quando o manifest não declara ``language_embedding``."""
    run_dir = tmp_path / "run"
    frame_dir = run_dir / "frames" / "frame-000"
    frame_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"config": {}}), encoding="utf-8")
    observation = frame_dir / "observation.json"
    observation.write_text("{}", encoding="utf-8")
    assert _language_embedding_config(observation) is None


# A configuração completa precisa sair exatamente como a run declarou, para
# que a referência de fusão carregue a identidade correta do espaço CLIP.
def test_language_embedding_config_le_manifest_completo(tmp_path: Path) -> None:
    """Lê backend, checkpoint, dimensão e normalização do manifest."""
    run_dir = tmp_path / "run"
    frame_dir = run_dir / "frames" / "frame-000"
    frame_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "config": {
            "language_embedding": {
                "backend": "clip", "checkpoint": "vit-b-32", "dimension": 4, "normalize": True,
            }
        }
    }), encoding="utf-8")
    observation = frame_dir / "observation.json"
    observation.write_text("{}", encoding="utf-8")
    config = _language_embedding_config(observation)
    assert config == {"backend": "clip", "checkpoint": "vit-b-32", "dimension": 4, "normalize": True}


# A ausência do archive do frame é um erro acionável, e não uma referência
# vazia: a capability está ligada e o consumidor precisa saber exatamente
# onde a run falhou em publicar a evidência.
def test_language_embedding_reference_falha_sem_archive(tmp_path: Path) -> None:
    """Recusa construir a referência quando ``embeddings.npz`` não existe."""
    frame_dir = tmp_path / "frame-000"
    frame_dir.mkdir()
    config = {"backend": "clip", "checkpoint": "vit-b-32", "dimension": 4, "normalize": True}
    with pytest.raises(ValueError, match="embeddings.npz"):
        _language_embedding_reference("region-1", frame_dir, config)


# Com o archive presente, a referência carrega identidade estável de espaço,
# dimensão e produtor, resolvível de volta ao archive do frame.
def test_language_embedding_reference_e_resolucao_round_trip(tmp_path: Path) -> None:
    """Constrói a referência e resolve o vetor persistido pelo mesmo archive."""
    frame_dir = tmp_path / "frame-000"
    frame_dir.mkdir()
    np.savez_compressed(frame_dir / "embeddings.npz", **{"region-1": np.asarray([0.6, 0.8], dtype=np.float32)})
    config = {"backend": "clip", "checkpoint": "vit-b-32", "dimension": 2, "normalize": True}

    reference = _language_embedding_reference("region-1", frame_dir, config)
    assert reference.embedding_id == "region-1"
    assert reference.embedding_space == "clip:vit-b-32:language_aligned"
    assert reference.producer == "language_embedding:clip:vit-b-32"
    assert reference.dimension == 2
    assert reference.normalized is True

    vector = _resolve_language_embedding(reference)
    assert vector == pytest.approx((0.6, 0.8), abs=1e-6)
