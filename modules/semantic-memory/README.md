# Semantic Memory

`semantic-memory` indexa embeddings alinhados à linguagem já ligados a uma
geometria persistente e responde buscas textuais por similaridade de
cosseno, exata e determinística.

## Responsabilidades

- transportar `SemanticRetrievalEntry`: identidade, geometria, coordenadas e
  vetor de uma entrada indexável;
- buscar por cosseno entre uma query e entradas compatíveis, ordenadas por
  score decrescente e desempatadas por identidade estável;
- recusar comparar espaços de embedding ou produtores incompatíveis.

## Não-responsabilidades

- fundir múltiplas observações em um embedding por geometria
  (`semantic-fusion.fuse_language_embeddings`);
- persistir ou reabrir vetores em disco (`semantic-map`);
- projetar texto em um espaço de embedding (aplicação, via um encoder
  concreto como CLIP);
- decidir qual capability está disponível para um mapa (`query-engine`).

## Por que o módulo existe

A primeira versão de busca por vocabulário aberto sobre o mapa precisa ser
reproduzível e auditável antes de precisar de escala. Um índice exato — sem
ANN, sem aproximação — garante que o mesmo par (mapa, query) sempre produz o
mesmo resultado, e mantém a superfície pequena o bastante para não esconder
bugs de comparação entre espaços incompatíveis.

## Fronteira pública

```text
SemanticRetrievalEntry  entrada indexável ligada a uma geometria persistente
VectorSearchResult      entrada e score de um match
search_entries          busca por cosseno entre entradas compatíveis
```

`search_entries` recusa a busca (não devolve lista vazia) quando nenhuma
entrada é compatível com o `embedding_space`/`producer` da query, quando a
dimensão da query diverge das entradas, ou quando a query tem norma zero.
Um índice vazio, por outro lado, devolve tupla vazia: ausência de conteúdo
não é a mesma coisa que incompatibilidade de espaço.
