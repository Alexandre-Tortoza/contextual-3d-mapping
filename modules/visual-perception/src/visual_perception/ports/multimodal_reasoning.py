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
from visual_perception.domain.region_reasoning import RegionReasoningRequest


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

    # Roda um prompt estruturado sobre uma região específica, a partir das
    # views mask-aware daquela região e do contexto de cena estruturado.
    # Chamado pelo estágio de region semantics (#165) uma vez por região
    # descoberta.
    #
    # A assinatura anterior recebia ``(image, mask_crop, scene_summary)``.
    # Ela foi substituída, e não duplicada, porque manter as duas formas
    # criaria um ponto de variação falso: existe um único consumidor de
    # produção, e duas entradas concorrentes tornariam impossível saber qual
    # evidência produziu um claim.
    def analyze_region(
        self, request: RegionReasoningRequest, config: MultimodalReasoningConfig
    ) -> dict[str, Any]:
        """Retorna uma resposta estruturada bruta descrevendo uma região (#165, #203).

        O implementador recebe as views já recortadas e distinguíveis por
        slot (``request.foreground_views`` e ``request.contextual_views``) e
        as claims de cena estruturadas. Ele não deve recortar a imagem por
        conta própria nem promover uma propriedade de cena a propriedade da
        região: essa é uma decisão de application, validada em
        ``parse_region_interpretation``.
        """
        ...
