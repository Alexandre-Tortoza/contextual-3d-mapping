# Query Engine

`query-engine` roteia uma consulta textual sobre um mapa para a capability
semântica disponível, e normaliza o resultado em um contract público
transportável (serializável em JSON), sem nunca declarar sucesso quando a
capability está ausente.

## Responsabilidades

- validar `MapQueryRequest` (identidade do mapa, texto e limite);
- rotear a consulta para `semantic-memory.search_entries` usando o encoder de
  texto injetado pela aplicação;
- normalizar matches em `MapQueryResult`, sem vazar tipos internos de
  `semantic-memory`;
- declarar `QueryOutcome("capability_unavailable", ...)` quando não há
  nenhuma entrada indexada, em vez de simular uma busca vazia bem-sucedida.

## Não-responsabilidades

- projetar texto em um espaço de embedding — nenhum backend de linguagem é
  importado aqui, nem mesmo por trás de uma interface; o encoder chega pronto
  como um `Callable[[str], tuple[float, ...]]`;
- indexar ou persistir vetores (`semantic-memory`, `semantic-map`);
- expor a consulta como serviço HTTP (`apps/map-explorer/api`).

## Por que o módulo existe

Nem toda run publica embeddings CLIP fundidos (a fusão de linguagem é
opt-in). A aplicação precisa de uma resposta única e estável para "este mapa
suporta busca por texto?" sem replicar essa checagem em cada consumidor —
web, CLI, ou um futuro cliente programático.

## Fronteira pública

```text
MapQueryRequest      pedido textual de busca em um mapa aberto
MapQueryResult        match compacto ligado a uma geometria persistente
QueryOutcome          resultado ou indisponibilidade explícita de capability
query_semantic_map    roteia a consulta e normaliza o resultado
```

Este módulo nunca importa `visual_perception`: quem verifica que o encoder de
texto injetado realmente reproduz o espaço/checkpoint/dimensão/normalização
declarados por um mapa é a aplicação que o chama
(`apps/map-explorer/api`), não `query-engine`.
