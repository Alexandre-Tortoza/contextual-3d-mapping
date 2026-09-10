# Comparação: 20260908T131207Z → 20260910T115810Z

| | baseline | candidato |
| --- | --- | --- |
| revisão | `03ec593` | `7001803` |
| prompt_version | `v6` | `v7` |
| perfil | `full` | `full` |
| pico de VRAM (GiB) | 4.57 | 4.57 |

> Comparação **estrutural**. Sem anotação humana revisada (#210/#211) nenhuma
> afirmação de acurácia semântica é possível, e nenhuma é feita aqui.

### corridor-02-000

| eixo | 20260908T131207Z | 20260910T115810Z |
| --- | ---: | ---: |
| mesma entrada (SHA-256) | — | sim |
| proposals | 75 | 75 |
| regiões canônicas | 40 | 40 |
| labels crus distintos | 8 | 8 |
| conceitos canônicos | n/a | 7 |
| label dominante | wall | wall |
| fração dominante | 0.45 | 0.42 |
| scene echo (1ª hipótese) | 0 | 0 |
| scene echo (qualquer) | n/a | 0 |
| confiança degenerada | sim | sim |
| claims não pontuadas | n/a | 0 |
| contradições conceito/natureza | n/a | 30 |
| regiões reconciliadas | n/a | 30 |
| hipóteses afirmadas concorrentes | n/a | 24 |
| sem suporte independente | n/a | 5 |
| identidade ambígua | n/a | 2 |
| grupos de superfície | n/a | 5 |
| regiões em grupo | n/a | 31 |
| grupos corroborados | n/a | 3 |
| relações geométricas | n/a | 210 |
| falhas de interpretação | 0 | 0 |
| falhas de evidência | 0 | 0 |
| falhas de sinal | n/a | 0 |
| falhas de relação | n/a | 0 |
| audit errors | 0 | 0 |
| audit warnings | 51 | 42 |
| latência (s) | 189 | 328 |
| chamadas de modelo (total) | 164 | 234 |

Relações semânticas: `covers` × 4, `part_of` × 4.
Custo dos estágios novos: hypothesis_support_text=30, region_refinement=24, semantic_relations=16.
Refinamento it0: 24 alvos, 24 reinterpretadas, evidência ['masked_subject', 'tight_crop', 'contextual_crop'] → ['foreground_dense', 'masked_subject', 'tight_crop', 'contextual_crop'], razões={'unsupported_primary': 5, 'small_region': 4, 'region_kind_contradiction': 20, 'insufficient_foreground': 2, 'competing_hypotheses': 2}.

### corridor-02-008

| eixo | 20260908T131207Z | 20260910T115810Z |
| --- | ---: | ---: |
| mesma entrada (SHA-256) | — | sim |
| proposals | 46 | 46 |
| regiões canônicas | 16 | 16 |
| labels crus distintos | 8 | 9 |
| conceitos canônicos | n/a | 7 |
| label dominante | plain surface | tree |
| fração dominante | 0.19 | 0.19 |
| scene echo (1ª hipótese) | 1 | 2 |
| scene echo (qualquer) | n/a | 2 |
| confiança degenerada | sim | não |
| claims não pontuadas | n/a | 0 |
| contradições conceito/natureza | n/a | 3 |
| regiões reconciliadas | n/a | 12 |
| hipóteses afirmadas concorrentes | n/a | 8 |
| sem suporte independente | n/a | 1 |
| identidade ambígua | n/a | 1 |
| grupos de superfície | n/a | 0 |
| regiões em grupo | n/a | 0 |
| grupos corroborados | n/a | 0 |
| relações geométricas | n/a | 44 |
| falhas de interpretação | 0 | 0 |
| falhas de evidência | 0 | 0 |
| falhas de sinal | n/a | 0 |
| falhas de relação | n/a | 0 |
| audit errors | 0 | 0 |
| audit warnings | 16 | 8 |
| latência (s) | 92 | 156 |
| chamadas de modelo (total) | 68 | 117 |

Relações semânticas: `covers` × 1, `part_of` × 1.
Custo dos estágios novos: hypothesis_support_text=30, region_refinement=8, semantic_relations=11.
Refinamento it0: 8 alvos, 8 reinterpretadas, evidência ['masked_subject', 'tight_crop', 'contextual_crop'] → ['foreground_dense', 'masked_subject', 'tight_crop', 'contextual_crop'], razões={'unsupported_primary': 1, 'small_region': 5, 'region_kind_contradiction': 3, 'competing_hypotheses': 4}.

### corridor-02-017

| eixo | 20260908T131207Z | 20260910T115810Z |
| --- | ---: | ---: |
| mesma entrada (SHA-256) | — | sim |
| proposals | 76 | 76 |
| regiões canônicas | 38 | 38 |
| labels crus distintos | 10 | 10 |
| conceitos canônicos | n/a | 9 |
| label dominante | ceiling | ceiling |
| fração dominante | 0.34 | 0.34 |
| scene echo (1ª hipótese) | 0 | 0 |
| scene echo (qualquer) | n/a | 0 |
| confiança degenerada | sim | sim |
| claims não pontuadas | n/a | 0 |
| contradições conceito/natureza | n/a | 27 |
| regiões reconciliadas | n/a | 27 |
| hipóteses afirmadas concorrentes | n/a | 24 |
| sem suporte independente | n/a | 6 |
| identidade ambígua | n/a | 3 |
| grupos de superfície | n/a | 5 |
| regiões em grupo | n/a | 26 |
| grupos corroborados | n/a | 3 |
| relações geométricas | n/a | 184 |
| falhas de interpretação | 0 | 0 |
| falhas de evidência | 0 | 0 |
| falhas de sinal | n/a | 0 |
| falhas de relação | n/a | 0 |
| audit errors | 0 | 0 |
| audit warnings | 49 | 43 |
| latência (s) | 175 | 307 |
| chamadas de modelo (total) | 156 | 223 |

Relações semânticas: `part_of` × 6, `covers` × 1.
Custo dos estágios novos: hypothesis_support_text=27, region_refinement=24, semantic_relations=16.
Refinamento it0: 24 alvos, 24 reinterpretadas, evidência ['masked_subject', 'tight_crop', 'contextual_crop'] → ['foreground_dense', 'masked_subject', 'tight_crop', 'contextual_crop'], razões={'unsupported_primary': 6, 'small_region': 6, 'region_kind_contradiction': 22, 'competing_hypotheses': 2, 'insufficient_foreground': 1}.

