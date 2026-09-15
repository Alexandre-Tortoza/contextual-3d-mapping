"""Tipos explícitos usados pela composição da CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Declara os recursos necessários para levar uma gravação até o viewer. Existe
# porque bags genéricos podem ser percebidos mesmo sem configuração geométrica.
@dataclass(frozen=True)
class DatasetProfile:
    """Configuração de composição conhecida para uma gravação.

    Argumentos:
        profile_id: identidade estável do perfil.
        bag: rosbag correspondente.
        camera_topic: tópico RGB calibrado.
        intrinsics: calibração intrínseca.
        extrinsics: calibração extrínseca.
        ground_truth: trajetória alternativa.
        sequence_masks: geometria válida da câmera para visual-perception.
        map_target: alvo Make que produz a geometria.
        context_target: alvo Make que compõe o artifact contextual.
    """

    profile_id: str
    bag: Path
    camera_topic: str
    intrinsics: Path
    extrinsics: Path
    ground_truth: Path
    sequence_masks: Path
    map_target: str
    context_target: str


# Representa as escolhas reproduzíveis da extração. É compartilhada entre o
# wizard e o subcomando para que ambos validem exatamente as mesmas invariantes.
@dataclass(frozen=True)
class ExtractionRequest:
    """Pedido de resolução e extração de uma janela RGB.

    Argumentos:
        bag: rosbag de origem.
        segment_id: identidade dos artifacts produzidos.
        start_s: início relativo ao primeiro frame RGB.
        duration_s: duração do trecho ou ``None`` para o bag inteiro.
        keyframe_interval_s: espaçamento da amostragem.
        all_frames: seleciona todas as imagens da janela.
        camera_topic: tópico explícito ou ``None`` para detecção.
        keyframe_offsets_s: posições explícitas após o início da janela, quando
            a campanha não usa espaçamento regular.
    """

    bag: Path
    segment_id: str
    start_s: float = 0.0
    duration_s: float | None = None
    keyframe_interval_s: float = 2.0
    all_frames: bool = False
    camera_topic: str | None = None
    keyframe_offsets_s: tuple[float, ...] | None = None


# Agrupa os artifacts produzidos na extração para evitar convenções de path
# implícitas entre os comandos seguintes.
@dataclass(frozen=True)
class ExtractionResult:
    """Artifacts reproduzíveis de uma extração.

    Argumentos:
        window: arquivo JSON da janela resolvida.
        frames_dir: diretório dos PNGs extraídos.
        frame_count: quantidade efetiva de frames.
    """

    window: Path
    frames_dir: Path
    frame_count: int


# Registra uma inconsistência acionável encontrada pela validação do projeto.
@dataclass(frozen=True)
class ValidationIssue:
    """Resultado individual da validação estrutural.

    Argumentos:
        level: severidade ``erro`` ou ``aviso``.
        message: diagnóstico em português.
        repairable: indica se ``--repair`` consegue corrigir com segurança.
    """

    level: str
    message: str
    repairable: bool = False


# Resume o custo conhecido antes de uma operação longa. Mantém valores
# ausentes explícitos quando ainda não existe histórico suficiente.
@dataclass(frozen=True)
class CostEstimate:
    """Estimativa contextual de uma execução.

    Argumentos:
        frame_count: quantidade prevista de frames.
        storage_bytes: espaço estimado ou ``None``.
        duration_s: tempo estimado ou ``None``.
    """

    frame_count: int
    storage_bytes: int | None = None
    duration_s: float | None = None
