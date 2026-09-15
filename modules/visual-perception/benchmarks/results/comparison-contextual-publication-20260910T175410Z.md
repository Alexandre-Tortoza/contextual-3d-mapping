# Comparação: 20260910T115810Z → 20260910T175410Z

| | baseline | candidato |
| --- | --- | --- |
| revisão | `7001803` | `3ac763a` |
| prompt_version | `v7` | `v7` |
| perfil | `full` | `full` |
| pico de VRAM (GiB) | 4.57 | 4.57 |

> Comparação **estrutural**. Sem anotação humana revisada (#210/#211) nenhuma
> afirmação de acurácia semântica é possível, e nenhuma é feita aqui.

### corridor-02-000

| eixo | 20260910T115810Z | 20260910T175410Z |
| --- | ---: | ---: |
| mesma entrada (SHA-256) | — | sim |
| proposals | 75 | 75 |
| regiões canônicas | 40 | 40 |
| regiões publicadas | n/a | 2 |
| contexto estrutural | n/a | 38 |
| labels crus distintos | 8 | 8 |
| conceitos canônicos | 7 | 7 |
| label dominante | wall | wall |
| fração dominante | 0.42 | 0.42 |
| scene echo (1ª hipótese) | 0 | 0 |
| scene echo (qualquer) | 0 | 0 |
| confiança degenerada | sim | sim |
| claims não pontuadas | 0 | 0 |
| contradições conceito/natureza | 30 | 30 |
| regiões reconciliadas | 30 | 30 |
| hipóteses afirmadas concorrentes | 24 | 24 |
| sem suporte independente | 5 | 5 |
| identidade ambígua | 2 | 2 |
| grupos de superfície | 5 | 5 |
| regiões em grupo | 31 | 31 |
| grupos corroborados | 3 | 3 |
| relações geométricas | 210 | 210 |
| falhas de interpretação | 0 | 0 |
| falhas de evidência | 0 | 0 |
| falhas de sinal | 0 | 0 |
| falhas de relação | 0 | 0 |
| audit errors | 0 | 0 |
| audit warnings | 42 | 42 |
| latência (s) | 328 | 321 |
| chamadas de modelo (total) | 234 | 234 |

Relações semânticas: `covers` × 4, `part_of` × 4.
Custo dos estágios novos: hypothesis_support_text=30, region_refinement=24, semantic_relations=16.
Refinamento it0: 24 alvos, 24 reinterpretadas, evidência ['masked_subject', 'tight_crop', 'contextual_crop'] → ['foreground_dense', 'masked_subject', 'tight_crop', 'contextual_crop'], razões={'unsupported_primary': 5, 'small_region': 4, 'region_kind_contradiction': 20, 'insufficient_foreground': 2, 'competing_hypotheses': 2}.

### corridor-02-008

| eixo | 20260910T115810Z | 20260910T175410Z |
| --- | ---: | ---: |
| mesma entrada (SHA-256) | — | sim |
| proposals | 46 | 46 |
| regiões canônicas | 16 | 16 |
| regiões publicadas | n/a | 13 |
| contexto estrutural | n/a | 3 |
| labels crus distintos | 9 | 9 |
| conceitos canônicos | 7 | 7 |
| label dominante | tree | tree |
| fração dominante | 0.19 | 0.19 |
| scene echo (1ª hipótese) | 2 | 2 |
| scene echo (qualquer) | 2 | 2 |
| confiança degenerada | não | não |
| claims não pontuadas | 0 | 0 |
| contradições conceito/natureza | 3 | 3 |
| regiões reconciliadas | 12 | 12 |
| hipóteses afirmadas concorrentes | 8 | 8 |
| sem suporte independente | 1 | 1 |
| identidade ambígua | 1 | 1 |
| grupos de superfície | 0 | 0 |
| regiões em grupo | 0 | 0 |
| grupos corroborados | 0 | 0 |
| relações geométricas | 44 | 44 |
| falhas de interpretação | 0 | 0 |
| falhas de evidência | 0 | 0 |
| falhas de sinal | 0 | 0 |
| falhas de relação | 0 | 0 |
| audit errors | 0 | 0 |
| audit warnings | 8 | 8 |
| latência (s) | 156 | 159 |
| chamadas de modelo (total) | 117 | 117 |

Relações semânticas: `covers` × 1, `part_of` × 1.
Custo dos estágios novos: hypothesis_support_text=30, region_refinement=8, semantic_relations=11.
Refinamento it0: 8 alvos, 8 reinterpretadas, evidência ['masked_subject', 'tight_crop', 'contextual_crop'] → ['foreground_dense', 'masked_subject', 'tight_crop', 'contextual_crop'], razões={'unsupported_primary': 1, 'small_region': 5, 'region_kind_contradiction': 3, 'competing_hypotheses': 4}.

### corridor-02-017

| eixo | 20260910T115810Z | 20260910T175410Z |
| --- | ---: | ---: |
| mesma entrada (SHA-256) | — | sim |
| proposals | 76 | 76 |
| regiões canônicas | 38 | 38 |
| regiões publicadas | n/a | 4 |
| contexto estrutural | n/a | 34 |
| labels crus distintos | 10 | 10 |
| conceitos canônicos | 9 | 9 |
| label dominante | ceiling | ceiling |
| fração dominante | 0.34 | 0.34 |
| scene echo (1ª hipótese) | 0 | 0 |
| scene echo (qualquer) | 0 | 0 |
| confiança degenerada | sim | sim |
| claims não pontuadas | 0 | 0 |
| contradições conceito/natureza | 27 | 27 |
| regiões reconciliadas | 27 | 27 |
| hipóteses afirmadas concorrentes | 24 | 24 |
| sem suporte independente | 6 | 6 |
| identidade ambígua | 3 | 3 |
| grupos de superfície | 5 | 5 |
| regiões em grupo | 26 | 26 |
| grupos corroborados | 3 | 3 |
| relações geométricas | 184 | 184 |
| falhas de interpretação | 0 | 0 |
| falhas de evidência | 0 | 0 |
| falhas de sinal | 0 | 0 |
| falhas de relação | 0 | 0 |
| audit errors | 0 | 0 |
| audit warnings | 43 | 43 |
| latência (s) | 307 | 310 |
| chamadas de modelo (total) | 223 | 223 |

Relações semânticas: `part_of` × 6, `covers` × 1.
Custo dos estágios novos: hypothesis_support_text=27, region_refinement=24, semantic_relations=16.
Refinamento it0: 24 alvos, 24 reinterpretadas, evidência ['masked_subject', 'tight_crop', 'contextual_crop'] → ['foreground_dense', 'masked_subject', 'tight_crop', 'contextual_crop'], razões={'unsupported_primary': 6, 'small_region': 6, 'region_kind_contradiction': 22, 'competing_hypotheses': 2, 'insufficient_foreground': 1}.

