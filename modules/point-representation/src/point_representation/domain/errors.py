"""Erros nomeados da fronteira de `point-representation`.

Issues: #3 (validação de entrada), #4 (seleção de canais).

Erros de fronteira deste módulo são exceptions nomeadas, e não ``ValueError``
genérico, para que um chamador distinga "nuvem malformada" de "canal
solicitado ausente" sem inspecionar a mensagem de texto.
"""

from __future__ import annotations


# Sinaliza que uma nuvem de pontos não satisfaz a política de entrada exigida
# pelo chamador (ex: vazia quando não permitido). Distinta de um erro de
# construção de ``PointCloud``: a nuvem em si é estruturalmente válida, só não
# é aceitável neste ponto do pipeline.
class PointCloudValidationError(ValueError):
    """Uma nuvem de pontos estruturalmente válida viola a política de entrada do chamador."""


# Sinaliza que a seleção de canais de entrada do encoder pediu um canal que a
# nuvem não carrega. Existe separado de ``PointCloudValidationError`` porque a
# causa é uma incompatibilidade de configuração, não um dado malformado.
class FeatureSelectionError(ValueError):
    """Um canal de entrada configurado não está disponível na nuvem fornecida."""
