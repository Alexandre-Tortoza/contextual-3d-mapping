# Consolidação de Runs Contextuais

`semantic-map` produz um artifact derivado para cada conjunto de runs
publicadas sobre a mesma nuvem de origem. A identidade é o hash canônico de
`map_id`, `map_frame`, `source.sha256` e `source.point_count`; igualdade de nome
de mapa não é suficiente. Runs de trechos diferentes carregam recortes
diferentes da mesma nuvem e continuam fundíveis, porque cada `geometry_id`
preserva o índice do ponto na origem. Um mesmo `geometry_id` com coordenadas
diferentes entre fontes é recusado.

Com um fundo global (slice geométrico da mesma origem), a geometria do mapa
consolidado é o fundo mais os pontos com contexto das runs; sem fundo, é a união
dos pontos das runs.

Cada ponto recebe no máximo um voto de cada run. O label desse voto é a decisão
já fundida dentro da run de origem. Antes da contagem, labels são normalizados e
agrupados por contenção contígua de tokens, de modo que `pallet` e `wooden
pallet` pertencem à mesma família. A saída mantém os dois labels crus nas
contribuições para auditoria.

Uma família com mais da metade dos votos observados vence. Caso nenhuma família
atinja maioria estrita, o maior `ranking_key` de `semantic-fusion` decide o
label e o resultado declara `quality_tiebreak`. A ausência de observação e
hipóteses sem label não contam como discordância.

Observações e regiões são namespaceadas por `run_id`. As previews continuam
nas pastas imutáveis das runs e são referenciadas por URLs públicas, evitando
duplicar imagens no artifact consolidado.
