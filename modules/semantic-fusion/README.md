# Semantic Fusion

`semantic-fusion` decide qual classificação um ponto do mapa persistente carrega quando várias observações visuais o classificam, e preserva todas as propostas concorrentes.

> Documentação detalhada: [`docs/`](docs/README.md).

## Responsabilidades

- receber contribuições semânticas ancoradas na mesma identidade de geometria;
- escolher um label primário por uma regra explícita e determinística;
- medir a concordância entre as observações contribuintes;
- preservar cada contribuinte com sua observação, região e confiança de origem.

## Não-responsabilidades

- projeção, visibilidade e oclusão (`sensor-association`);
- persistência das entidades semânticas resultantes, capacidade futura de `semantic-map`;
- calibração da confiança dos claims (`visual-perception`).

## Por que o módulo existe

Com múltiplos keyframes cobrindo o mesmo trecho, a mesma superfície pode receber propostas diferentes. Sem uma regra explícita, a última gravação venceria por ordem de execução e o mapa deixaria de ser determinístico.

## Regra de fusão

`fuse_point_contributions` ordena as contribuições por:

1. confiança calibrada, quando o produtor a fornece;
2. confiança bruta declarada, como alternativa;
3. fração de evidência visual favorável (`visual_support`);
4. qualidade geométrica da máscara (`region_quality`);
5. instante da observação e identificador da região, como desempate determinístico.

Os sinais de suporte entram como desempate, e não como uma soma ponderada que fingiria uma calibração ainda inexistente.

`agreement` é a fração de contribuições que concordam com o label primário. Nenhuma contribuição é descartada.

Na implementação atual, essa fração é calculada **depois** da escolha e não
participa do ranking. Uma observação errada com maior confiança pode vencer
mesmo contra várias outras. Confiança pouco discriminativa transfere a escolha
aos desempates; `visual_support` e `region_quality` também não comprovam
grounding espacial. Não há tracking nem consenso temporal robusto nesta regra.

O mapping-runtime agora cria contribuições apenas de associações fortes sobre
footprints grounded. Tentativas de boundary e falhas de grounding ficam no
artifact, fora deste ranking. Essa condição melhora a entrada individual;
não transforma concordância temporal em validação da máscara.

## Suporte espacial

`measure_spatial_support` mede quanto a vizinhança geométrica concorda com o label de cada ponto em uma grade de voxels configurável.

Uma vizinhança com poucos vizinhos rotulados devolve `None`, e não zero. Isso preserva a diferença entre ausência de evidência e evidência contrária.

No trecho de 30 s do corridor-02, contra o subconjunto comprovadamente incoerente, os sinais observados foram:

| Sinal | Marca | Alcança dos incoerentes | Razão |
| --- | --- | --- | --- |
| suporte espacial < 34% | 20,3% | 35,2% | 1,73 |
| concordância entre keyframes < 50% | 12,3% | 34,4% | 2,80 |
| ambos abaixo de 50% | 9,7% | 33,2% | 3,42 |

Esses números são evidência de um experimento específico, não thresholds universais.

## Fronteira pública

```text
SemanticContribution     proposta de uma observação para um ponto
FusedPointContext        label primário, concordância e contribuintes preservados
fuse_point_contributions
LabelledPoint            ponto do mapa com o label que carrega
SpatialNeighbourhood     parâmetros da grade de vizinhança
measure_spatial_support
```

O label primário é uma escolha para consumo downstream, não um apagamento das alternativas.