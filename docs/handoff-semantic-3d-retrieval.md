# Handoff: embeddings semânticos 3D e busca textual

## Objetivo e decisões fechadas

Implementar a cadeia #222 → #223 → #140 → #116 → #117 → #224 sem revisão
humana. A fonte canônica é o embedding CLIP alinhado à linguagem, não a feature
densa DINO. A primeira interface é o map-explorer por API HTTP local; vetores
não podem ser enviados ao browser. A busca usa cosseno exato e determinístico.

## Estado já implementado

- `semantic-fusion` ganhou `LanguageEmbeddingReference`,
  `FusedLanguageEmbedding` e `fuse_language_embeddings`.
- A fusão resolve vetores por callback, exige espaço/dimensão/produtor
  idênticos, ordena por observação/região/referência e normaliza a média.
- Peso explícito: confiança calibrada, confiança bruta, `visual_support`,
  `region_quality`, ou `1.0`; todos zero viram pesos uniformes.
- `sensor-association` carrega `language_embedding_reference` de
  `VisualRegionEvidence` até `PointVisualAssociation`.
- `mapping-runtime` já extrai `language_embedding_ref` da região visual e a
  grava no registro associado.
- `semantic-map/embeddings.py` escreve/relê embeddings por `geometry_id` em
  NPZ e retorna metadata compacta, sem vetor no JSON.
- Foram criados os módulos mínimos `semantic-memory` (entrada e busca exata)
  e `query-engine` (request, resultado, outcome e rota semântica).
- `python -m compileall` passou para os módulos novos e alterados.

## Trabalho restante, nesta ordem

1. Corrigir/adicionar testes unitários de `fuse_language_embeddings`: uma e
   várias fontes, ordem de entrada, pesos ausentes/zero, dimensão incompatível,
   resolver ausente e vetor nulo. Não alterar o caminho DINO já existente.
2. No `mapping-runtime`, construir `LanguageEmbeddingReference` para cada
   contribuição forte. A referência deve usar `language_embedding_ref`,
   `embeddings.npz` do frame, e metadata CLIP lida da configuração/manifest da
   run. Archive ou metadata ausente deve falhar com erro acionável quando a
   capability for habilitada; ausência de ref na região permanece explícita.
3. Ligar a fusão CLIP ao `semantic-map`: consolidar contribuições por
   `geometry_id`, escrever `semantic-embeddings.npz`, publicar no artifact
   somente `archive_uri`, espaço, dimensão, produtor, normalização,
   contributors e pesos. Atualização deve reabrir todos os contributors e
   recalcular, nunca agregar médias previamente fundidas.
4. Criar fixtures e testes de round-trip do mapa, incluindo mapa sem embeddings
   e metadata/archive inválidos.
5. Criar `apps/map-explorer/api` como serviço HTTP local. Não há FastAPI
   instalada; usar `http.server` ou adicionar uma dependência explícita.
   A factory deve receber entradas e encoder injetado, expor health e
   `POST /v1/maps/{map_id}/query`, e devolver somente `QueryOutcome`
   serializável.
6. O adapter de aplicação usa o `LanguageAlignedEncoder` CLIP existente para
   texto e verifica espaço, checkpoint, dimensão e normalização antes de
   chamar `query_engine`. O `query-engine` não pode importar visual-perception.
7. No web, configurar a URL da API local, adicionar campo de texto, loading,
   erro/capacidade indisponível, lista ordenada e ação que encontra
   `geometry_id` no mapa e chama a seleção/foco. Não carregar NPZ no browser.
8. Rodar `ruff check` somente em Python, pytest dos módulos e `npm test` no
   web. Depois atualizar READMEs em português e fechar #222, #223, #140,
   #116, #117 e #224 apenas se todos os critérios estiverem cobertos.

## Cuidados com o worktree

O worktree já estava sujo antes desta tarefa, principalmente em
visual-perception, point-representation, map-explorer e runtime. Não reverter
nem formatar arquivos não relacionados. Alterações da milestone anterior de
coerência visual e diagnóstico geométrico coexistem com esta cadeia.
