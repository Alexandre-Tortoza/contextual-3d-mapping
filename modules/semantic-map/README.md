# Semantic Map

`semantic-map` consolida evidência semântica de runs contextuais publicadas
sobre a mesma geometria persistente. As runs de origem permanecem imutáveis; o
resultado é um artifact derivado, reproduzível e rastreável até cada run,
observação e região que o sustentam.

## Fronteira pública

```text
PublishedContextRun       run contextual publicada e sua proveniência
ConsolidatedContextMap    resultado versionado da consolidação
geometry_fingerprint      identidade canônica da geometria compartilhada
consolidate_context_runs  funde evidência entre runs compatíveis
write_semantic_embedding_archive  persiste embeddings CLIP fundidos por geometry_id
read_semantic_embedding           reabre um vetor persistido, validando dimensão
```

## Regra de consolidação

Cada run contribui no máximo um voto por ponto. Variantes textuais como
`pallet` e `wooden pallet` são agrupadas por contenção de tokens. Uma maioria
estrita vence; sem maioria, o melhor claim individual decide pelo ranking de
`semantic-fusion`, e o resultado é marcado como `quality_tiebreak`.

Somente runs com o mesmo `map_id`, `map_frame` e fingerprint de todos os
pontos podem ser consolidadas. O módulo não registra mapas diferentes nem
infere alinhamentos espaciais.

## Persistência de embeddings semânticos

`write_semantic_embedding_archive` grava os vetores de `FusedLanguageEmbedding`
(fundidos por `semantic_fusion.fuse_language_embeddings`) em um archive NPZ
indexado por `geometry_id`, e devolve metadata compacta — espaço, dimensão,
produtor, normalização, contribuintes e pesos efetivos — sem nenhum vetor no
JSON do mapa. `read_semantic_embedding` reabre um vetor por `geometry_id` e
valida a dimensão declarada contra o vetor persistido.

Um mapa sem nenhum embedding CLIP fundido não é um erro: o archive
simplesmente não é escrito, e a metadata publicada fica vazia. Quem consome
essa metadata (`map-explorer/api`) trata a ausência como capability
indisponível, não como falha.
