"""Configuração de algoritmo de `point-representation`.

Issue: #4.

A configuração de seleção de canais é intencionalmente aberta a qualquer
nome de canal: o módulo não declara um enum fechado de canais suportados,
porque "outro canal registrado" (#4) é exatamente o ponto de variação — um
produtor pode anexar um canal novo a ``PointCloud.channels`` sem que este
módulo precise mudar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from point_representation.domain.spatial_bounds import AxisAlignedBounds


# Declara quais canais auxiliares de ``PointCloud.channels`` compõem a
# entrada do encoder, e em que ordem. A ordem é significativa: ela determina
# a ordem de concatenação em ``EncoderInput.features`` (#4), então precisa
# ser estável para que a dimensão de saída seja determinística.
@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Quais canais auxiliares (além das coordenadas) alimentam o encoder.

    Argumentos:
        channels: nomes de canal, na ordem de concatenação. Vazio para
            entrada somente-coordenadas.
    """

    channels: tuple[str, ...] = ()


# Configura o recorte espacial determinístico de uma nuvem (#5). O subsample
# opcional pós-recorte exige ``seed`` explícito porque a issue #5 exige que a
# mesma configuração produza sempre o mesmo resultado — uma amostragem sem
# seed não seria reproduzível entre execuções.
@dataclass(frozen=True)
class CropConfig:
    """Limites do recorte e, opcionalmente, um subsample determinístico pós-recorte.

    Argumentos:
        bounds: caixa alinhada aos eixos que define a região retida.
        max_points: se definido, subamostra o recorte para no máximo esta
            contagem de pontos, escolhidos com ``seed``.
        seed: obrigatório quando ``max_points`` é definido.
    """

    bounds: AxisAlignedBounds
    max_points: int | None = None
    seed: int | None = None

    # Rejeita uma configuração que pediria subsample sem uma seed, já que o
    # resultado deixaria de ser determinístico — o requisito central da #5.
    def __post_init__(self) -> None:
        """Valida que ``max_points`` positivo sempre venha acompanhado de ``seed``."""
        if self.max_points is not None:
            if self.max_points <= 0:
                raise ValueError(f"CropConfig.max_points must be positive, got {self.max_points}.")
            if self.seed is None:
                raise ValueError("CropConfig.seed is required when max_points is set.")


# Configura o downsampling por voxel (#6). Um ``voxel_size_m`` não-positivo
# não define uma grade válida, então é rejeitado aqui em vez de produzir um
# comportamento silenciosamente degenerado no downsampling.
@dataclass(frozen=True)
class VoxelDownsampleConfig:
    """Tamanho da célula de voxel, em metros, usado para agrupar pontos.

    Argumentos:
        voxel_size_m: aresta da célula de voxel; deve ser positiva.
    """

    voxel_size_m: float

    def __post_init__(self) -> None:
        """Valida que ``voxel_size_m`` é positivo."""
        if self.voxel_size_m <= 0.0:
            raise ValueError(f"VoxelDownsampleConfig.voxel_size_m must be positive, got {self.voxel_size_m}.")


# Configura a normalização de coordenadas (#7). Os dois eixos de
# configuração são independentes: desligar os dois é o caso "normalização
# desabilitada" exigido pela issue, sem precisar de uma terceira flag.
@dataclass(frozen=True)
class NormalizationConfig:
    """Quais componentes da normalização de coordenadas aplicar.

    Argumentos:
        center: subtrai o centróide das coordenadas antes de escalar.
        scale: divide as coordenadas centralizadas pela maior magnitude
            absoluta observada, com fallback seguro no caso degenerado.
    """

    center: bool = True
    scale: bool = True


# Configura o baseline supervisionado que destila features de um professor
# visual em uma representação 3D por ponto. Existe para validar o caminho de
# supervisão 2D->3D antes de introduzir um backbone de maior custo ou runtime.
@dataclass(frozen=True)
class RidgeDistillationConfig:
    """Parâmetros reproduzíveis para a destilação linear por ponto.

    Argumentos:
        ridge_regularization: penalidade L2 aplicada aos pesos, exceto ao
            bias. Zero seleciona mínimos quadrados sem regularização.
    """

    ridge_regularization: float = 1e-6

    # Rejeita uma penalidade negativa, que não corresponde a uma regressão
    # ridge e pode tornar o sistema normal mal condicionado.
    def __post_init__(self) -> None:
        """Valida que a regularização ridge não seja negativa."""
        if not math.isfinite(self.ridge_regularization) or self.ridge_regularization < 0.0:
            raise ValueError("RidgeDistillationConfig.ridge_regularization must be finite and non-negative.")
