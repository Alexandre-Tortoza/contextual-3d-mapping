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
```

## Regra de consolidação

Cada run contribui no máximo um voto por ponto. Variantes textuais como
`pallet` e `wooden pallet` são agrupadas por contenção de tokens. Uma maioria
estrita vence; sem maioria, o melhor claim individual decide pelo ranking de
`semantic-fusion`, e o resultado é marcado como `quality_tiebreak`.

Somente runs com o mesmo `map_id`, `map_frame` e fingerprint de todos os
pontos podem ser consolidadas. O módulo não registra mapas diferentes nem
infere alinhamentos espaciais.
