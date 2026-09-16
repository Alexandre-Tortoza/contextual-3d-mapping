"""Primitivas e composição de renderização usadas pelos artifacts de debug.

Fronteira pública mínima: os consumidores importam diretamente de
``visual_perception.rendering.layers`` (formas e primitivas de desenho) e
``visual_perception.rendering.overlay`` (mapeamento de tipos de domínio para
formas desenháveis). Este pacote existe para que ``debug_artifacts.py`` e
outros artifacts de inspeção do módulo não dependam de código de
``benchmarks/``, que não é instalável como dependência de outro consumidor.
"""

from __future__ import annotations
