"""Integra a execução canônica de percepção com o mapping-runtime.

Issue: #179.

``mapping-runtime`` (#106) ainda não está implementado. Este módulo mostra e
testa o formato de integração pretendido: o runtime chama apenas a fronteira
pública de ``visual_perception`` (``run_canonical_pipeline`` e seus tipos
públicos), nunca um caminho de implementação privado, e recebe falhas do
módulo como um diagnóstico explícito em vez de deixar vazar uma exception
crua de tipo específico de backend.
"""

from __future__ import annotations

from dataclasses import dataclass

from visual_perception import ImageObservation, ImagePayload, ModuleConfig, PerceptionPorts, VisualObservation
from visual_perception.application.pipeline import run_canonical_pipeline
from visual_perception.domain.errors import VisualPerceptionError


# Representa uma falha do módulo já traduzida para o formato que o
# mapping-runtime espera consumir — nunca a exception crua do backend.
# Existe para que o runtime nunca precise conhecer tipos de exception
# internos de visual-perception, só este diagnóstico estável.
@dataclass(frozen=True)
class RuntimeDiagnostic:
    """Falha de módulo exposta ao runtime, nunca uma exception crua de backend."""

    observation_id: str
    stage: str
    message: str


# Executa o pipeline canônico exatamente como o mapping-runtime deve chamá-lo:
# converte qualquer VisualPerceptionError capturada em um RuntimeDiagnostic
# em vez de propagar a exception, para que o runtime nunca precise tratar
# tipos de erro internos do módulo.
def run_visual_perception_for_runtime(
    image: ImageObservation, payload: ImagePayload, config: ModuleConfig, ports: PerceptionPorts
) -> VisualObservation | RuntimeDiagnostic:
    """Executa o pipeline canônico da forma como ``mapping-runtime`` deve chamá-lo."""
    try:
        result = run_canonical_pipeline(image, payload, config, ports)
    except VisualPerceptionError as error:
        return RuntimeDiagnostic(
            observation_id=image.observation_id, stage="visual_perception", message=str(error)
        )
    return result.observation
