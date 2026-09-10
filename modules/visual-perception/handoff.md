# Handoff: comparar backends sobre a arquitetura contextual já fixa

## Objetivo do próximo agente

A arquitetura de composição de evidência está implementada, integrada ao caminho canônico
e validada em execução real. O que falta agora é o que só faz sentido **depois** dela:
comparar backends sobre uma arquitetura que não muda mais entre os braços da comparação.

Não trate a rodada anterior como concluída porque os testes passam. Três defeitos reais
dela só apareceram na execução real, com a suíte inteira verde. Leia
`docs/visual-context-sota-review.md` §10 antes de decidir o que fazer.

## Estado recebido

- Data do handoff: 2026-09-10.
- Branch: `main`.
- Revisões desta rodada: `39e6888`, `3c546a7`, `4ce4e3c`, `89ed955`.
- `prompt_version` atual: `v7`. Os prompts de cena e de região são byte-idênticos aos do
  `v6`; o `v7` acrescentou o prompt de **relação**.
- `schema_version` da `VisualObservation`: **3**. Ela ganhou `entity_hypotheses` e os
  sinais de suporte por claim; payloads v1 e v2 continuam legíveis.
- Layout de artifacts por frame: **`frames/2`**, com `embeddings.npz`.
- Suíte: 491 verificações determinísticas mais os testes de GPU, todas verdes por
  `make verify`.

### O que passou a existir

| capacidade | estágio | issue |
| --- | --- | --- |
| sinal de suporte por hipótese, do canal alinhado à linguagem | `application/hypothesis_support.py` | #214 |
| refinamento seletivo dirigido por razão explícita | `application/refinement.py` | #204 |
| reconciliação contextual intra-frame e grupos de superfície | `application/reconciliation.py` | #205 |
| relações semânticas candidatas sobre pares priorizados | `application/semantic_relations.py` | #206 |
| coerência determinística conceito ↔ natureza | `domain/structural_consistency.py` | #216 |
| comparação de backend em um fator só | `benchmarks/validate_reference_pipeline.py` (`--reasoning-checkpoint`) | #218 |
| embeddings expostos e persistidos por referência | `benchmarks/frame_artifacts.py` | #217 |
| integração de tudo em uma ordem determinística | `application/pipeline.py` | #207 |

## Os três frames vinculantes

Inalterados. Use exatamente estes IDs, nesta ordem, com estes hashes:

| Frame | SHA-256 |
| --- | --- |
| `corridor-02-000` | `4a69f4d01bed3534b7581aa6199a45fb3be7b91a405a58bc44af29261f3cf1e6` |
| `corridor-02-008` | `2a0db0d9ee1ad0e3d3f05785c028b0c63221194e7589c55c6b0ad1c73601b78b` |
| `corridor-02-017` | `f7ea622db269c79244af5dfc5ae66bd5d1fbe2b5babee60cb51d71acccd5a20a` |

## Baselines que não devem ser apagadas

```text
benchmarks/results/samples/20260908T131207Z/   baseline sem estágios contextuais (03ec593)
benchmarks/results/samples/20260910T115810Z/   REFERÊNCIA desta rodada (7001803)
benchmarks/results/samples/20260909T205243Z/   16 keyframes, corroboração em escala (478fe63)
benchmarks/results/samples/20260909T135428Z/   superado, registro do defeito de escalonamento
benchmarks/results/comparison-contextual-20260910T115810Z.md
```

A comparação é reproduzível:

```bash
python benchmarks/compare_runs.py \
  --baseline benchmarks/results/samples/20260908T131207Z \
  --candidate benchmarks/results/samples/<novo>
```

> **`20260909T135428Z` está superado.** Ele foi produzido com `scene_conditioned` no
> escalonamento de refinamento, configuração revertida por reintroduzir o vazamento de
> cena. Fica versionado como registro do defeito; **não cite como resultado**.

O run de referência é o `20260910T115810Z` (revisão `7001803`): três frames vinculantes,
hashes conferidos, sem fallback, sem OOM, zero erro de audit, e **zero eco de cena
introduzido por refinamento** nos três frames.

O `20260909T205243Z` (16 keyframes, mesma configuração) corrobora em escala: **zero eco em
462 regiões**.

## Trabalho recomendado, em ordem

1. **#218 — comparar Qwen2.5-VL-3B contra Qwen3-VL 2B e 4B.** É a comparação com melhor
   relação custo/informação disponível: os três checkpoints já estão no cache local, a
   versão instalada de `transformers` suporta a família, e não há download nenhum. Fixe
   tudo menos o checkpoint — prompt, views, temperatura, tetos dos estágios contextuais —
   e meça os eixos da issue. Uma troca de prompt junto com a troca de modelo torna as
   duas ininterpretáveis.

2. **#219 e #220** dependem de você aceitar as licenças de `facebook/dinov3-*` e
   `facebook/sam3` no Hugging Face e configurar um token. O bloqueio não é técnico: o
   código da versão instalada de `transformers` já suporta os dois.

3. **Calibrar os tetos dos estágios novos.** `refinement.max_regions_per_iteration` e
   `semantic_relations.max_pairs` foram escolhidos como orçamento plausível, **não**
   medidos. O custo hoje é 1,74x da baseline em latência, com VRAM inalterada.

4. **#210/#211 continuam sendo o bloqueio real.** Enquanto não houver anotação humana
   revisada, nenhuma métrica de acurácia, ECE, Brier ou taxa de alucinação pode ser
   publicada, e nenhuma comparação de backend pode afirmar superioridade semântica —
   apenas diferença estrutural.

## Uma armadilha do ambiente, não do código

`grep` neste shell é uma **função** instalada pelo Claude Code que roteia para `claude -G`.
Ela repassa poucas flags; `grep -vE` cai no `claude`, que lê o `-v` como `--version`,
imprime a versão e sai. Num pipe, isso fecha o cano e mata o processo da esquerda por
SIGPIPE — foi assim que um braço de benchmark morreu deixando só um diretório vazio, com
exit code 0.

Ao encanar a saída de um run longo, use `/usr/bin/grep` ou `command grep`, ou não encane
nada: o arquivo de log já guarda tudo.

## Três armadilhas que esta rodada pagou para descobrir

Elas estão medidas em `docs/known-limitations.md` e em `docs/visual-context-sota-review.md`
§3.3 e §10. Repetir qualquer uma custaria um run inteiro.

1. **Uma capacidade que nenhum consumidor lê não existe.** O refinamento acrescentava
   claims que `primary_label_claim` nunca devolvia, então gastava chamadas de VLM e não
   mudava nada observável. Antes de declarar um estágio pronto, verifique quem **lê** a
   saída dele.
2. **Um prompt que informa a resposta recebe a resposta de volta.** Entregar a fração de
   contenção ao modelo fez ele devolver `inside` em 6 de 16 pares; sem ela, 0 de 16. Se um
   campo do prompt é sinônimo de uma resposta possível, ele não é contexto — é gabarito.
3. **Um contexto global no prompt de uma região vira identidade da região.** Vale para
   texto (medido na `#202`) e para pixels (medido agora): a view de cena no escalonamento
   fez 6 de 16 regiões adotarem a claim `layout` como label. O frame que expôs isso foi o
   único que **não** é um corredor.

## Critérios de encerramento do próximo ciclo

- run real na configuração atual, com os três hashes vinculantes;
- comparação estrutural gerada por `benchmarks/compare_runs.py` e versionada;
- limitações e resultados negativos registrados, inclusive os que contrariarem a hipótese
  de partida;
- somente issues com todos os critérios comprovados fechadas;
- nenhuma afirmação de acurácia enquanto `#210`/`#211` estiverem abertas.
