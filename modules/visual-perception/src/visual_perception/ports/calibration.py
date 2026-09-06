"""Fronteira substituível de calibração de claims semânticos.

Issue: #196.

A regra que converte suporte estruturado em um score confiável é um ponto
de variação real: ela depende de dados medidos, muda de versão junto com o
artifact que a define, e precisa ser trocável entre experimentos sem tocar
no pipeline. Por isso ela é um port, e não uma função fixa.

Implementações vivem em ``application/semantic_calibration.py``. Nenhuma
delas pode inventar um score: quando a evidência não sustenta uma
calibração, o contract exige abstenção explícita
(:class:`~visual_perception.domain.semantic_support.SemanticSupport` no
estado ``abstained``), nunca um número plausível.
"""

from __future__ import annotations

from typing import Protocol

from visual_perception.domain.semantic_support import SemanticSupport, SupportInputs


# Port que desacopla o pipeline da regra concreta de calibração (tabela de
# confiabilidade medida, temperature scaling, ou uma regra futura). Existe
# para que a mesma fronteira sirva a um calibrador treinado, a um fake
# determinístico de teste e ao caminho "sem calibração", que apenas preserva
# o score bruto sem promovê-lo a calibrado.
class ClaimCalibrator(Protocol):
    """Converte suporte estruturado em um :class:`SemanticSupport` decidido."""

    # Identifica a versão da regra aplicada. Entra em cada SemanticSupport
    # calibrado para que uma mudança de calibração seja rastreável nos
    # resultados que ela produziu.
    @property
    def version(self) -> str:
        """A versão do contract de calibração aplicada por este calibrador."""
        ...

    # Identifica o artifact concreto de dados usado, quando existe. Existe
    # separado de ``version`` porque duas execuções da mesma versão de
    # contract podem usar tabelas medidas diferentes.
    @property
    def artifact_id(self) -> str | None:
        """O digest ou caminho do artifact de calibração, ou ``None`` sem artifact."""
        ...

    # Ponto de entrada único do port: decide entre pontuar, se abster ou
    # reportar falha, a partir dos sinais estruturados de uma claim.
    def calibrate(self, inputs: SupportInputs) -> SemanticSupport:
        """Decide o suporte de uma claim: calibrado, bruto, abstenção ou falha."""
        ...
