# Consolidação de Runs Contextuais

`semantic-map` produz um artifact derivado para cada conjunto de runs
publicadas que compartilha exatamente a mesma geometria. A identidade é o hash
canônico de `map_id`, `map_frame`, `geometry_id` e `coordinates_m`; igualdade
de nome de mapa não é suficiente.

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
