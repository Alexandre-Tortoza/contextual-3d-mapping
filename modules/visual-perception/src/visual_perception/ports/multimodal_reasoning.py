"""Fronteira de raciocínio multimodal para interpretação em nível de cena e de região.

Issue: #189 (shape do adapter real). Usada pelos estágios de application em
#164 (scene) e #165 (region). O port retorna uma resposta estruturada bruta;
o parsing de schema e a validação semântica ficam na camada de application,
não em código de transporte específico de backend (ver escopo da #189).
"""

from __future__ import annotations

from typing import Any, Protocol

from visual_perception.config import MultimodalReasoningConfig
from visual_perception.domain.image_payload import ImagePayload


# Port que desacopla o pipeline do VLM (vision-language model) concreto usado
# para raciocínio multimodal. Existe para que backends diferentes (ou um
# fake determinístico em testes) possam responder aos mesmos prompts
# estruturados de scene/region sem vazar seu schema de transporte específico.
class MultimodalReasoner(Protocol):
    """Executa prompts estruturados em nível de cena e de região em um backend."""

    # Roda um prompt estruturado sobre a cena inteira. Chamado pelo estágio
    # de scene context do pipeline (#164) uma vez por observação de imagem.
    def analyze_scene(
        self, image: ImagePayload, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna uma resposta estruturada bruta descrevendo a cena inteira.

        Respostas malformadas/não parseáveis são responsabilidade do estágio
        de application (#164); este método deve levantar
        :class:`~visual_perception.domain.errors.BackendExecutionError` apenas
        para falhas de backend/transporte.
        """
        ...

    # Roda um prompt estruturado sobre uma região específica, opcionalmente
    # usando o resumo de cena já produzido por ``analyze_scene`` como
    # contexto adicional. Chamado pelo estágio de region semantics (#165)
    # uma vez por região descoberta.
    def analyze_region(
        self,
        image: ImagePayload,
        mask_crop: ImagePayload,
        scene_summary: str | None,
        config: MultimodalReasoningConfig,
    ) -> dict[str, Any]:
        """Retorna uma resposta estruturada bruta descrevendo uma região (#165)."""
        ...
