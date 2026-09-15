"""Leitura e validação de campanhas contextuais de trechos do corridor-02."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ENVIRONMENTS = frozenset({"indoor", "outdoor"})
REASONING_BACKENDS = frozenset({"qwen_vl", "gemini_robotics_er"})


# Representa um cenário temporal da campanha. Existe para que a seleção de
# ambientes seja auditável e não fique dispersa em comandos de shell.
@dataclass(frozen=True)
class CampaignSegment:
    """Trecho temporal identificado e classificado por ambiente."""

    segment_id: str
    environment: str
    start_s: float


# Mantém a política inteira em uma fronteira tipada consumível pela CLI e por
# testes, antes que qualquer rosbag seja aberta ou um artifact seja escrito.
@dataclass(frozen=True)
class ContextCampaign:
    """Configuração reproduzível de uma campanha de percepção e contexto."""

    campaign_id: str
    duration_s: float
    keyframe_offsets_s: tuple[float, ...]
    voxel_size_m: float
    max_viewer_points: int
    segments: tuple[CampaignSegment, ...]
    #: Backend do reasoner da percepção; ``None`` usa o Qwen local da referência.
    reasoning_backend: str | None = None


# Lê a configuração versionada e rejeita combinações que tornariam a campanha
# ambígua ou impossível de reproduzir. Cardinalidades específicas de uma campanha
# (quantos trechos, quantos outdoor) são verificadas no teste daquela campanha.
def load_campaign(path: Path) -> ContextCampaign:
    """Carrega e valida uma campanha TOML.

    Argumentos:
        path: arquivo TOML da campanha.
    Retorna:
        campanha pronta para resolução de janelas.
    Levanta:
        ValueError: quando trechos, ambientes, offsets ou backend forem inválidos.
    """
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    offsets = tuple(float(value) for value in payload["keyframe_offsets_s"])
    duration_s = float(payload["duration_s"])
    segments = tuple(
        CampaignSegment(str(item["id"]), str(item["environment"]), float(item["start_s"]))
        for item in payload["segments"]
    )
    reasoning_backend = payload.get("reasoning_backend")
    if not segments or len({item.segment_id for item in segments}) != len(segments):
        raise ValueError("a campanha deve declarar ao menos um segmento, com ids únicos.")
    if any(item.environment not in ENVIRONMENTS for item in segments):
        raise ValueError(f"o ambiente de cada segmento deve ser um de {sorted(ENVIRONMENTS)}.")
    if duration_s <= 0 or any(item.start_s < 0 for item in segments):
        raise ValueError("duração deve ser positiva e inícios dos segmentos não negativos.")
    if not offsets or list(offsets) != sorted(set(offsets)) or offsets[0] < 0 or offsets[-1] > duration_s:
        raise ValueError("os offsets de keyframe devem ser crescentes, únicos e dentro da janela.")
    if reasoning_backend is not None and reasoning_backend not in REASONING_BACKENDS:
        raise ValueError(f"reasoning_backend deve ser um de {sorted(REASONING_BACKENDS)}.")
    return ContextCampaign(
        campaign_id=str(payload["campaign_id"]), duration_s=duration_s,
        keyframe_offsets_s=offsets, voxel_size_m=float(payload["voxel_size_m"]),
        max_viewer_points=int(payload["max_viewer_points"]), segments=segments,
        reasoning_backend=reasoning_backend,
    )
