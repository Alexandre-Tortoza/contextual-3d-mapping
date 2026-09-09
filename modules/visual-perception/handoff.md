# Handoff: comparar backends sobre a arquitetura contextual já fixa

## Objetivo do próximo agente

A arquitetura de composição de evidência está implementada, integrada ao caminho canônico
e validada em execução real. O que falta agora é o que só faz sentido **depois** dela:
comparar backends sobre uma arquitetura que não muda mais entre os braços da comparação.

Não trate a rodada anterior como concluída porque os testes passam. Três defeitos reais
dela só apareceram na execução real, com a suíte inteira verde. Leia
`docs/visual-context-sota-review.md` §10 antes de decidir o que fazer.

## Estado recebido

- Data do handoff: 2026-09-09.
- Branch: `main`.
- Revisões desta rodada: `39e6888`, `3c546a7`, `4ce4e3c`.
- `prompt_version` atual: `v7`. Os prompts de cena e de região são byte-idênticos aos do
  `v6`; o `v7` acrescentou o prompt de **relação**.
- `schema_version` da `VisualObservation`: **3**. Ela ganhou `entity_hypotheses` e os
  sinais de suporte por claim; payloads v1 e v2 continuam legíveis.
- Layout de artifacts por frame: **`frames/2`**, com `embeddings.npz`.
- Suíte: 487 verificações determinísticas mais os testes de GPU, todas verdes por
  `make verify`.

### O que passou a existir

| capacidade | estágio | issue |
| --- | --- | --- |
| sinal de suporte por hipótese, do canal alinhado à linguagem | `application/hypothesis_support.py` | #214 |
| refinamento seletivo dirigido por razão explícita | `application/refinement.py` | #204 |
| reconciliação contextual intra-frame e grupos de superfície | `application/reconciliation.py` | #205 |
| relações semânticas candidatas sobre pares priorizados | `application/semantic_relations.py` | #206 |
| coerência determinística conceito ↔ natureza | `domain/structural_consistency.py` | #216 |
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
benchmarks/results/samples/20260909T135428Z/   arquitetura contextual (1270e53)
benchmarks/results/comparison-contextual-20260909T135428Z.md
```

A comparação é reproduzível:

```bash
python benchmarks/compare_runs.py \
  --baseline benchmarks/results/samples/20260908T131207Z \
  --candidate benchmarks/results/samples/<novo>
```

> **Ressalva sobre `20260909T135428Z`.** Ele foi produzido com `scene_conditioned` no
> escalonamento de refinamento, e essa configuração foi **revertida depois** por
> reintroduzir o vazamento de cena (limitação 7 de `known-limitations.md`). Os eixos de
> geometria, sinal, grupos e relações continuam válidos; os labels das 8 regiões afetadas
> de `corridor-02-008` não. Um run com a configuração atual ainda não foi produzido —
> **é a primeira coisa a fazer**, e a GPU estava ocupada por outro processo quando esta
> rodada terminou.

## Trabalho recomendado, em ordem

1. **Reproduzir o run dos três frames na configuração atual.** Ele é o baseline de tudo
   que vem depois, e a única coisa que falta para fechar esta rodada:

   ```bash
   ./.venv/bin/python benchmarks/validate_reference_pipeline.py \
     --context-profile full \
     --frame-id corridor-02-000 --frame-id corridor-02-008 --frame-id corridor-02-017
   ```

   Confirme no manifest: `git_revision` contendo o comportamento avaliado, os três
   SHA-256, ausência de `feature_fallback_reason`, zero `audit_error_count`, e
   `contextual.scene_echo_any_assertion` igual a zero nos três frames.

2. **#218 — comparar Qwen2.5-VL-3B contra Qwen3-VL 2B e 4B.** É a comparação com melhor
   relação custo/informação disponível: os três checkpoints já estão no cache local, a
   versão instalada de `transformers` suporta a família, e não há download nenhum. Fixe
   tudo menos o checkpoint — prompt, views, temperatura, tetos dos estágios contextuais —
   e meça os eixos da issue. Uma troca de prompt junto com a troca de modelo torna as
   duas ininterpretáveis.

3. **#219 e #220** dependem de você aceitar as licenças de `facebook/dinov3-*` e
   `facebook/sam3` no Hugging Face e configurar um token. O bloqueio não é técnico: o
   código da versão instalada de `transformers` já suporta os dois.

4. **Calibrar os tetos dos estágios novos.** `refinement.max_regions_per_iteration` e
   `semantic_relations.max_pairs` foram escolhidos como orçamento plausível, **não**
   medidos. O custo hoje é 1,90x da baseline em latência, com VRAM inalterada.

5. **#210/#211 continuam sendo o bloqueio real.** Enquanto não houver anotação humana
   revisada, nenhuma métrica de acurácia, ECE, Brier ou taxa de alucinação pode ser
   publicada, e nenhuma comparação de backend pode afirmar superioridade semântica —
   apenas diferença estrutural.

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
