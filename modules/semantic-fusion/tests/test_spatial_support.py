"""Testes da medição de suporte espacial de labels contra a vizinhança."""

from __future__ import annotations

import pytest
from semantic_fusion import LabelledPoint, SpatialNeighbourhood, measure_spatial_support


# Distribui pontos ao longo de uma linha com espaçamento menor que a aresta do
# voxel, de modo que todos caiam na mesma vizinhança 3×3×3.
def _linha(labels: list[str], espacamento: float = 0.1) -> list[LabelledPoint]:
    """Cria pontos colineares e vizinhos entre si, com os labels informados."""
    return [
        LabelledPoint(f"p{index}", (index * espacamento, 0.0, 0.0), label)
        for index, label in enumerate(labels)
    ]


# Uma superfície homogênea é o caso em que o label está bem sustentado, e a
# medição não pode penalizá-la.
def test_vizinhanca_homogenea_da_suporte_total() -> None:
    """Confere suporte 1,0 quando todos os vizinhos concordam."""
    suporte = measure_spatial_support(_linha(["parede"] * 6))
    assert set(suporte.values()) == {1.0}


# É o caso que motiva o módulo: um ponto cercado de vizinhos que afirmam outra
# coisa, como um ponto atrás da parede que herdou o label da superfície da frente.
def test_ponto_isolado_entre_discordantes_tem_suporte_baixo() -> None:
    """Confere suporte próximo de zero para um label contradito pela vizinhança."""
    pontos = _linha(["parede", "parede", "porta", "parede", "parede", "parede"])
    suporte = measure_spatial_support(pontos)
    assert suporte["p2"] == 0.0
    assert all(suporte[f"p{index}"] > 0.5 for index in (0, 1, 3, 4, 5))


# Uma fronteira legítima entre duas superfícies tem suporte parcial nos dois
# lados. Ela precisa ficar distinguível do caso anterior, senão a medição
# apagaria toda transição real entre superfícies.
def test_fronteira_entre_superficies_mantem_suporte_parcial() -> None:
    """Confere que os dois lados de uma fronteira ficam acima do caso contradito."""
    pontos = _linha(["chao"] * 3 + ["parede"] * 3)
    suporte = measure_spatial_support(pontos)
    assert all(0.3 <= suporte[f"p{index}"] <= 0.7 for index in range(6))


# Um ponto que ninguém contradisse não foi refutado, e tratá-lo como sem
# suporte confundiria ausência de evidência com evidência contrária.
def test_vizinhanca_esparsa_deixa_o_suporte_indefinido() -> None:
    """Confere que poucos vizinhos produzem ``None`` em vez de zero."""
    distantes = [
        LabelledPoint("a", (0.0, 0.0, 0.0), "parede"),
        LabelledPoint("b", (10.0, 0.0, 0.0), "parede"),
    ]
    suporte = measure_spatial_support(distantes)
    assert suporte == {"a": None, "b": None}


# A aresta governa o que conta como vizinho, e precisa mudar o resultado de
# forma previsível para poder ser calibrada por dataset.
def test_aresta_do_voxel_governa_o_alcance_da_vizinhanca() -> None:
    """Confere que uma aresta menor isola pontos antes vizinhos."""
    pontos = _linha(["parede"] * 6, espacamento=0.5)
    largo = measure_spatial_support(pontos, SpatialNeighbourhood(voxel_edge_m=1.0))
    estreito = measure_spatial_support(
        pontos, SpatialNeighbourhood(voxel_edge_m=0.05, minimum_neighbours=1)
    )
    assert largo["p2"] == 1.0
    assert estreito["p2"] is None


# Sem determinismo, dois runs sobre o mesmo mapa produziriam atenuações
# diferentes no viewer e nenhuma comparação seria reproduzível.
def test_medicao_e_deterministica_e_independente_da_ordem() -> None:
    """Confere que a ordem de entrada não altera o suporte medido."""
    pontos = _linha(["parede", "parede", "porta", "parede", "chao", "parede"])
    direto = measure_spatial_support(pontos)
    invertido = measure_spatial_support(list(reversed(pontos)))
    assert direto == invertido


# Identidades repetidas indicariam composição inválida, e silenciá-las faria a
# medição sobrescrever um ponto com outro.
def test_identidades_repetidas_falham_na_fronteira() -> None:
    """Confere a validação de identidades duplicadas."""
    repetido = [
        LabelledPoint("p", (0.0, 0.0, 0.0), "parede"),
        LabelledPoint("p", (0.1, 0.0, 0.0), "parede"),
    ]
    with pytest.raises(ValueError, match="unique geometry ids"):
        measure_spatial_support(repetido)


# Parâmetros impossíveis precisam falhar antes de qualquer medição.
def test_vizinhanca_invalida_falha_na_construcao() -> None:
    """Confere a validação da aresta e do mínimo de vizinhos."""
    with pytest.raises(ValueError, match="voxel_edge_m"):
        SpatialNeighbourhood(voxel_edge_m=0.0)
    with pytest.raises(ValueError, match="minimum_neighbours"):
        SpatialNeighbourhood(minimum_neighbours=0)
