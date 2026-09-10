# #218 — raciocínio multimodal sobre a arquitetura contextual fixa

Comparação de três checkpoints nos três frames vinculantes, com **um único fator
variando**. Verificado nos manifests: o único campo de configuração que difere entre os
braços é `multimodal_reasoning.checkpoint`. Frames, geometria, views, prompt, temperatura,
quantização 4-bit e os tetos de todos os estágios contextuais são idênticos.

| braço | run | revisão |
| --- | --- | --- |
| referência, Qwen2.5-VL-3B-Instruct | `samples/20260910T115810Z/` | `7001803` |
| Qwen3-VL-2B-Instruct | `samples/20260910T121258Z/` | `7001803` |
| Qwen3-VL-4B-Instruct | `samples/20260910T122608Z/` | `7001803` |

Os três: 94 regiões, zero falha de interpretação, zero falha de evidência, zero erro de
audit, sem fallback, pico de 4,57 GiB. A geometria é idêntica nos três, como tem de ser —
ela é decidida antes de o VLM ser consultado.

> **Sem afirmação de acurácia.** As `#210`/`#211` continuam abertas. Tudo abaixo é
> diferença estrutural medida. Onde um eixo *parece* indicar qualidade, o texto diz
> explicitamente se ele é circular.

## A tabela, e por que ela engana se lida de cima para baixo

| eixo | 2.5-VL-3B | 3-VL-2B | 3-VL-4B |
| --- | ---: | ---: | ---: |
| latência total | 792 s | 765 s | **523 s** |
| chamadas de modelo | 574 | 524 | 442 |
| audit warnings | 93 | 87 | **10** |
| sinais `supports` | 162 | 157 | **0** |
| sinais `contradicts` | 63 | 28 | **0** |
| sinais `indistinguishable` | 57 | 73 | **0** |
| regiões sem suporte independente | 12 | 0 | **0** |
| identidade ambígua | 6 | 8 | **0** |
| contradições conceito/natureza | 60 | 73 | **8** |
| grupos de superfície (corroborados) | 10 (6) | 7 (3) | 12 (**11**) |
| relações semânticas | 17 | 38 | 38 |
| `part_of` / `covers` / `attached_to` | 11 / 6 / 0 | 2 / 21 / 15 | 14 / 4 / 19 |
| claims não pontuadas | 0 | 0 | 0 |

Lida ingenuamente, essa tabela diz que o 4B ganha em quase tudo. **Ela não diz isso.**

## O que explica metade da coluna do 4B

O Qwen3-VL-4B devolve `alternatives: []` em **40 de 40 regiões** de `corridor-02-000`. Os
outros dois devolvem uma ou duas alternativas em **40 de 40**.

```text
2.5-VL-3B   regiões com alternativa: 40/40   (16 com uma, 24 com duas)
3-VL-2B     regiões com alternativa: 40/40   (16 com uma, 24 com duas)
3-VL-4B     regiões com alternativa:  0/40
```

Sem hipótese concorrente, o canal de alinhamento não tem o que arbitrar. Todos os sinais
saem `unavailable`, e quatro linhas da tabela viram zero **por construção, não por
qualidade**:

- `supports`/`contradicts`/`indistinguishable` = 0 porque nada foi medido;
- "regiões sem suporte independente" = 0 porque a condição exige sinais medidos;
- "identidade ambígua" = 0 pelo mesmo motivo;
- os *warnings* despencam porque `contradictory_claims` e os dois códigos baseados em
  sinal não podem disparar;
- as chamadas de refinamento caem de 56 para 12 porque duas das razões — primária sem
  suporte e hipóteses concorrentes — nunca ocorrem.

Para este módulo isso **não é uma melhoria: é uma regressão**. A arquitetura inteira é
construída sobre preservar a dúvida que o produtor declara, e um modelo que não declara
dúvida nenhuma desliga o único canal de evidência independente que existe.

## O eixo que não é circular, e nele o 4B ganha de verdade

A coerência entre conceito e natureza é calculada por uma regra determinística do domínio,
que não consulta modelo nenhum. É o único eixo de qualidade não circular disponível hoje,
e nele a diferença é grande:

| | 2.5-VL-3B | 3-VL-2B | 3-VL-4B |
| --- | ---: | ---: | ---: |
| contradições conceito/natureza (94 regiões) | 60 | 73 | **8** |
| `RegionKind` em `corridor-02-000` | `thing` 33, `stuff` 7 | `thing` 39, `stuff` 1 | **`stuff` 30, `thing` 6, `part` 4** |

O 4B é o único que usa o contract como ele foi desenhado: paredes, tetos e pisos saem como
`stuff`, e ele é o único dos três que emite `part` alguma vez. Os outros dois respondem
`thing` para quase tudo — o erro que a limitação 5 documenta desde a `#202`.

Isso é uma diferença real e verificável sem anotação humana.

## O que não dá para decidir

- **`contradicts` caindo de 63 para 28 no 2B não significa que ele acerta mais.** O canal
  independente é o CLIP, e um modelo cujos labels são mais genéricos e mais frequentes no
  treino do CLIP pontua melhor no alinhamento sem ser mais correto. É o viés de escala e
  especificidade descrito em arXiv:2607.10993, e é indistinguível de acerto sem ground
  truth;
- **a distribuição de predicados muda qualitativamente entre os três**, e nenhum critério
  disponível ordena as três. `attached_to` sai de 0 no 3B para 15 no 2B e 19 no 4B;
  `covers` vai de 6 para 21 e volta para 4. São grafos diferentes sobre os mesmos pixels;
- **`occludes` não foi emitido por nenhum dos três**, em 94 regiões. Somado ao run de 16
  keyframes (1 ocorrência em 462 regiões), a evidência acumulada é que este predicado não
  é usado na prática por esta família de modelos.

## Conclusão

**Nenhuma troca de backend é recomendada por esta medição**, e a razão não é empate: é que
os dois candidatos trocam uma propriedade boa por uma ruim, em direções opostas.

O Qwen3-VL-4B é 34% mais rápido, declara a natureza das regiões corretamente sete vezes
mais, corrobora 11 dos 12 grupos que propõe, e **cala sobre a própria incerteza**. Adotá-lo
como está trocaria o canal de evidência independente por uma tabela de números melhores.

O caminho que a medição sugere, e que **não** foi seguido aqui porque mudaria dois fatores
de uma vez, é: alterar o prompt para exigir alternativas, versionar isso como
`prompt_version`, e então repetir esta comparação. Se o 4B mantiver a coerência estrutural
*e* passar a declarar dúvida, aí existe uma troca justificada.

Até lá, a referência continua Qwen2.5-VL-3B-Instruct.

## Reprodução

```bash
python benchmarks/validate_reference_pipeline.py --context-profile full \
  --reasoning-checkpoint Qwen/Qwen3-VL-4B-Instruct \
  --frame-id corridor-02-000 --frame-id corridor-02-008 --frame-id corridor-02-017

python benchmarks/compare_runs.py \
  --baseline benchmarks/results/samples/20260910T115810Z \
  --candidate benchmarks/results/samples/<braço>
```
