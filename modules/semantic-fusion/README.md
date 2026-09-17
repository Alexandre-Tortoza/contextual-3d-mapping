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

## Coerência visual densa

`PointVisualFeature` transporta a feature amostrada no pixel associado, com
espaço de embedding, dimensão implícita no vetor, produtor e referência ao
artifact. `fuse_point_contributions` compara somente features com mesmo espaço,
produtor e dimensão, por similaridade cosseno. O resultado é um
`VisualCoherence` separado de confiança, `agreement` e suporte espacial.

As políticas são `disabled`, `diagnostic` (default) e
`downrank_contradictions`. A política diagnóstica não altera o ranking; a de
ablação só rebaixa contribuições sem feature quando há contradição comprovada.
Features ausentes, vetores de norma zero e espaços incompatíveis ficam
explicitamente `unavailable`.

Na implementação atual, essa fração é calculada **depois** da escolha e não
participa do ranking. Uma observação errada com maior confiança pode vencer
mesmo contra várias outras. Confiança pouco discriminativa transfere a escolha
aos desempates; `visual_support` e `region_quality` também não comprovam
grounding espacial. Não há tracking nem consenso temporal robusto nesta regra.

O mapping-runtime agora cria contribuições apenas de associações fortes sobre
footprints grounded. Tentativas de boundary e falhas de grounding ficam no
artifact, fora deste ranking. Essa condição melhora a entrada individual;
não transforma concordância temporal em validação da máscara.

## Fusão de embeddings alinhados à linguagem

`fuse_language_embeddings` agrega os embeddings CLIP (`LanguageEmbeddingReference`)
de uma mesma geometria persistente em um único `FusedLanguageEmbedding`,
normalizado por L2. Exige espaço, dimensão, produtor e normalização idênticos
entre todas as referências; o resolver é injetado pelo chamador (`mapping-runtime`),
porque este módulo nunca abre o archive onde o vetor mora.

O peso de cada contribuição segue a mesma precedência de `fuse_point_contributions`
— confiança calibrada, confiança bruta, `visual_support`, `region_quality`, e
`1.0` como último recurso — e cai para peso uniforme quando todos os sinais são
zero ou ausentes. A ordenação por observação/região/`embedding_id` garante que a
mesma entrada produza sempre a mesma saída, independente da ordem de chegada.

Esta fusão é a fonte canônica para busca por vocabulário aberto (`semantic-memory`,
`query-engine`): a feature DINO densa (`fuse_point_features`) continua servindo
coerência visual multi-view, não busca textual.

## Suporte espacial

`measure_spatial_support` mede quanto a vizinhança geométrica concorda com o label de cada ponto em uma grade de voxels configurável.

Uma vizinhança com poucos vizinhos rotulados devolve `None`, e não zero. Isso preserva a diferença entre ausência de evidência e evidência contrária.

## Validação geométrica pós-2D

`measure_geometric_support` mede planicidade, extensão, consistência de normal
e protrusão sobre a geometria persistente. A medição recebe a natureza ampla
declarada pelo produtor (`surface`, `object`, `part` ou `unknown`) e devolve
`coherent`, `contradictory` ou `unresolved`, sempre com motivo e medidas.

O default é `diagnostic`: um objeto projetado sobre um plano contínuo é marcado
como contraditório, mas seu claim 2D bruto permanece intacto. Boundary,
geometria esparsa e natureza desconhecida permanecem `unresolved`. As políticas
ativas permanecem bloqueadas pelo contract até que a referência humana seja
revisada e a avaliação comprove o ganho.

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
LanguageEmbeddingReference  referência rastreável a um embedding CLIP
FusedLanguageEmbedding      embedding CLIP fundido de uma geometria persistente
fuse_language_embeddings
LabelledPoint            ponto do mapa com o label que carrega
SpatialNeighbourhood     parâmetros da grade de vizinhança
measure_spatial_support
GeometricSemanticPoint  ponto persistente e natureza ampla declarada
GeometricSupport        diagnóstico de compatibilidade geométrica por ponto
measure_geometric_support
```

O label primário é uma escolha para consumo downstream, não um apagamento das alternativas.
