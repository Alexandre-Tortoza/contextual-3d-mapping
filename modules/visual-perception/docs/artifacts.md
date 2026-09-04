# Artifacts

## Serialização canônica (issue #172)

[`infrastructure/serialization.py`](../src/visual_perception/infrastructure/serialization.py)
serializa uma `VisualObservation` para um dict JSON-able simples e de volta, fazendo
round-trip sem perda de informação:

- masks são embutidas como booleanos compactos com run-length-encoding — pequenas,
  exatas e autocontidas, já que a própria geometria de uma região é necessária para
  interpretar a observação de qualquer forma;
- embeddings visuais/de linguagem são referenciados só por id
  (`visual_embedding_ref`/`language_embedding_ref`); os vetores em si não fazem parte da
  observação canônica (veja [integração de persistência](#persistência));
  `Evidence.artifact` referencia evidência crua de modelo da mesma forma, através da
  `SourceArtifactReference` compartilhada;
- `schema_version` e `coordinate_convention` viajam com cada payload;
  `UnsupportedSchemaVersionError` falha de forma previsível em uma versão que este
  módulo não consegue ler.

## Persistência

`infrastructure/integration/persistence_integration.py` (#180) define o
`EvidencePersistencePort` que este módulo precisa da persistência do repositório (#112)
— um par opaco `put(id, payload) -> reference` / `get(reference) -> payload` — e:

- `persist_observation` / `reload_observation` fazem round-trip de uma
  `VisualObservation` completa;
- `persist_visual_embeddings` / `persist_language_embeddings` persistem os embeddings
  pooled que uma execução do pipeline produziu (eles vivem fora da observação canônica
  em si, referenciados só por id).

Nenhum client ou path específico de armazenamento aparece em nenhum contract público de
módulo; os testes exercitam essa fronteira com
`infrastructure/fakes/fake_evidence_store.InMemoryEvidenceStore`.

O guia [integration.md](integration.md) posiciona essa fronteira no fluxo de uma
aplicação. A semântica dos campos serializados continua sendo definida pelos contracts
em [api-contracts.md](api-contracts.md), não pelo backend de persistência.

## Artifacts do cache de estágio

Veja [execution.md](execution.md#cache-de-estágio-issue-170): `application/cache.StageCache`
escreve um registro JSON por `(stage_name, fingerprint)` sob um diretório de cache
escolhido pelo chamador, independente do formato de serialização canônico acima.
