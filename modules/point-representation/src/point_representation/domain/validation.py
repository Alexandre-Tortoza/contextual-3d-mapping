"""Fronteira de validação de entrada para nuvens de pontos.

Issue: #3.

Invariantes estruturais (forma, finitude, alinhamento de canal) já são
impostos por ``PointCloud.__post_init__`` — uma instância inválida não existe.
O que resta como decisão de fronteira, e não invariante estrutural, é a
política sobre nuvens vazias: um chamador de treino tipicamente não quer uma
amostra vazia, enquanto um chamador de inferência em lote pode precisar
tolerá-la (ex: uma tile sem pontos válidos). Esta função é o único ponto onde
essa política é aplicada, para que ela nunca fique implícita em cada
chamador.
"""

from __future__ import annotations

from point_representation.domain.errors import PointCloudValidationError
from point_representation.domain.point_cloud import PointCloud


# Aplica a política de entrada do pipeline sobre uma nuvem já estruturalmente
# válida. Existe como a fronteira que qualquer estágio de pré-processamento ou
# inferência deve atravessar antes de tocar os dados (#3).
def validate_point_cloud(cloud: PointCloud, *, allow_empty: bool = False) -> PointCloud:
    """Valida ``cloud`` contra a política de entrada do pipeline e a retorna.

    Argumentos:
        cloud: nuvem já estruturalmente válida (ver :class:`PointCloud`).
        allow_empty: quando ``False`` (padrão), rejeita nuvens com zero pontos.
    Retorna:
        a mesma ``cloud``, inalterada.
    Levanta:
        PointCloudValidationError: se a nuvem estiver vazia e ``allow_empty``
            for ``False``.
    """
    if not allow_empty and len(cloud) == 0:
        raise PointCloudValidationError(
            "PointCloud has zero points and allow_empty is False."
        )
    return cloud
