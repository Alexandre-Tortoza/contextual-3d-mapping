# Map Explorer API

Serviço HTTP local que expõe busca textual sobre um mapa contextual
publicado (#224). Composição executável: escolhe a leitura concreta do
artifact em disco e o backend CLIP real, e conecta essas escolhas ao
serviço HTTP genérico — nenhum vetor cruza para o browser, só
`QueryOutcome` serializável.

## Rodando

```bash
python -m map_explorer_api.app <map_id> <caminho/para/context.json> [--host 127.0.0.1] [--port 8765]
```

`map_id` deve ser o `map_id` que o próprio artifact declara (o mesmo valor
que o web configura como identidade do mapa aberto).

## Rotas

- `GET /health` — diagnóstico único do serviço.
- `POST /v1/maps/{map_id}/query` — corpo `{"text": str, "limit"?: int}`,
  resposta `{"status", "reason", "results"}` (o mesmo formato de
  `query_engine.QueryOutcome`).

Um mapa sem `semantic_embeddings` publicado responde
`capability_unavailable`, não um erro. Um `map_id` desconhecido responde
404 `map_not_found`.

## Composição

```text
map_index.py     adapta o artifact de mapa em SemanticRetrievalEntry
clip_encoder.py  verifica espaço/checkpoint/dimensão/normalização e projeta texto via visual-perception
service.py       handler HTTP genérico injetado com entradas e encoder
app.py           composição concreta: mapa em disco + backend CLIP real
```

`service.py` não sabe onde um mapa mora nem qual backend de texto roda — só
`app.py` decide isso. Um deployment com múltiplos mapas substituiria só
`app.py`.
