# Semantic Fusion

Esta documentação descreve a implementação local de `semantic-fusion`. A etapa correspondente na pipeline completa está em [`docs/19-semantic-fusion.md`](../../../docs/19-semantic-fusion.md).

## Papel no sistema

`semantic-fusion` recebe múltiplas contribuições semânticas já ancoradas na mesma geometria persistente e produz um contexto fundido sem apagar as evidências concorrentes.

```text
PointVisualAssociation de vários keyframes
    -> SemanticContribution[]
    -> semantic-fusion
        -> seleção determinística do label primário
        -> agreement entre observações
        -> suporte espacial 3D
        -> preservação dos contribuintes
    -> FusedPointContext
```

## Implementação atual

A implementação está dividida em:

- [`models.py`](../src/semantic_fusion/models.py), contracts públicos e validações;
- [`fusion.py`](../src/semantic_fusion/fusion.py), fusão multi-observação por ponto;
- [`spatial.py`](../src/semantic_fusion/spatial.py), suporte semântico na vizinhança geométrica.

## Fusão de contribuições

`fuse_point_contributions()` recebe propostas para um mesmo ponto e ordena as contribuições por sinais explícitos. Confiança calibrada tem precedência quando existe; confiança bruta é usada como fallback; sinais de suporte visual e qualidade geométrica participam como desempate; timestamp e identificador de região tornam o resultado determinístico.

O resultado contém:

- label primário;
- confiança do contribuinte vencedor;
- observação e região de origem;
- `agreement`, a fração de contribuições que concorda com o label primário;
- todas as contribuições ordenadas, preservadas para auditoria.

A seleção de um primário não transforma as demais hipóteses em erro nem remove sua proveniência.

## Suporte espacial

`measure_spatial_support()` mede a consistência local de labels em uma grade de voxels. O sinal é complementar ao `agreement` temporal/multi-view: uma classificação pode ser repetida por vários keyframes e ainda estar geometricamente deslocada.

Ausência de vizinhos suficientes é representada como ausência de evidência, e não como suporte zero.

## Estrutura atual

```text
modules/semantic-fusion/
├── README.md
├── docs/
│   └── README.md
├── src/
│   └── semantic_fusion/
│       ├── __init__.py
│       ├── models.py
│       ├── fusion.py
│       └── spatial.py
└── tests/
    ├── test_semantic_fusion.py
    └── test_spatial_support.py
```

## Fronteiras

O módulo assume que projeção, visibilidade e oclusão já foram resolvidas por `sensor-association`. Ele também não persiste entidades semânticas de alto nível, responsabilidade futura de `semantic-map`.

A calibração dos scores de percepção permanece com `visual-perception`; `semantic-fusion` consome a confiança calibrada quando ela existe, sem inventar uma calibração local.

## Relação com a documentação principal

- [`docs/18-point-visual-association.md`](../../../docs/18-point-visual-association.md), origem das associações geométrico-visuais;
- [`docs/19-semantic-fusion.md`](../../../docs/19-semantic-fusion.md), posição da fusão no fluxo completo;
- [`docs/20-semantic-map.md`](../../../docs/20-semantic-map.md), consumidor planejado do contexto fundido.

Decisões sobre ranking, concordância e suporte espacial pertencem a este diretório. O fluxo entre módulos permanece em [`docs/`](../../../docs/README.md).

## Próxima leitura

- [19. Semantic Fusion](../../../docs/19-semantic-fusion.md)
- [20. Semantic Map / Semantic Memory](../../../docs/20-semantic-map.md)
- [Pipeline end-to-end](../../../docs/README.md)