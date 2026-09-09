# Revisão do contexto visual 2D contra o estado da arte

Este documento é a **auditoria** que precede qualquer mudança de código nesta rodada. Ele
responde a uma pergunta só:

> a evidência contextual 2D que este módulo entrega hoje é boa o suficiente para que
> `sensor-association`, `semantic-fusion`, `semantic-map`, `semantic-memory`,
> `scene-graph` e `context-reasoning` façam o trabalho deles sem voltar a consultar o VLM?

A resposta curta é **não, e o motivo não é o modelo**. Os quatro backends selecionados
pelo benchmark #174 produzem evidência abundante que o pipeline calcula, paga em VRAM e
latência, e depois **descarta ou degenera**. As três medições centrais desta auditoria são:

| medição | valor | consequência |
| --- | ---: | --- |
| claims de label primário com `confidence == 0.90` exatamente | **165/165** | o score do produtor não carrega informação nenhuma |
| regiões com warning `contradictory_claims` | **163/165** | o único sinal que dirige refinamento é ruído estrutural |
| chamadas ao CLIP por frame cujo vetor nunca é lido | **121** | o canal independente de evidência é computado e jogado fora |

Nenhuma dessas três é um problema de checkpoint. As três são de **composição de
evidência**, que é a hipótese central desta revisão.

> **Escopo.** Tudo aqui termina em 2D. Projeção, identidade temporal, fusão, mapa,
> memória e reasoning de mapa continuam fora, e a seção
> [O que os papers fazem e não pertence aqui](#5-o-que-os-papers-fazem-e-não-pertence-a-este-módulo)
> torna essa fronteira explícita.

---

## 0. Base de evidência desta auditoria

Todas as contagens vêm de artifacts versionados, não de leitura de código.

| origem | o que é |
| --- | --- |
| `benchmarks/results/samples/20260908T131207Z/` | run real mais recente, revisão `03ec593`, 5 frames, 165 regiões canônicas, `prompt_version v6`, pico de 4,57 GiB |
| `benchmarks/evidence_signal_probe.py` | sonda reproduzível criada nesta rodada; mede os sinais de alinhamento e coerência descritos abaixo |
| `docs/known-limitations.md` | limitações já medidas em runs anteriores |

O run analisado cobre `corridor-02-000`, `-002`, `-008`, `-012` e `-017`. Os três frames
vinculantes do projeto (`000`, `008`, `017`) estão contidos nele.

Hardware: RTX 3060, 8 GiB. Nenhuma conclusão desta auditoria depende de hardware maior.

**Nenhuma métrica de acurácia semântica é publicada aqui.** As issues `#210`/`#211`
continuam abertas e o conjunto de referência continua `pending_review`; tudo abaixo é
estrutural, ou é um *proxy* explicitamente nomeado como tal.

---

## 1. Capacidades atuais

O que o módulo **de fato faz hoje**, verificado no código e confirmado nos artifacts:

| capacidade | onde vive | estado |
| --- | --- | --- |
| region discovery class-agnostic | `region_discovery_backend.py` (SAM ViT-H) | funcional |
| filtragem de validade (lente, rig, área) | `application/proposal_filtering.py` | funcional, auditável por `RejectedProposal` |
| merge cross-scale determinístico | `application/region_merge.py` | funcional, preserva `contributing_proposal_ids` |
| features densas em resolução elevada | `feature_extraction_backend.py` (DINOv2-base @448) | funcional, grade 32x24 medida |
| pooling mask-aware pixel-aligned | `application/pooling.py` | funcional, `support_ratio` reportado |
| quatro slots de evidência por região | `application/multi_context.py` | 165/165 regiões com 5/5 slots `available` |
| views em pixels distinguíveis por slot | `application/region_views.py` | funcional (`neutral_fill`, `contour`, crop cru) |
| contexto de cena **ambiental** | `application/scene_context.py` | funcional; inventário de objetos removido e recusado por contract |
| hipóteses de identidade primary/alternative | `application/region_semantics.py` | funcional, papel explícito, `raw_response_json` preservado |
| contract de suporte e abstenção | `domain/semantic_support.py` | contract completo, **sem dados** (calibração desligada) |
| relações geométricas 2D | `application/relation_generation.py` | funcional (`overlaps`, `contains`, `near`) |
| auditoria estrutural | `application/quality_audit.py` | funcional, mas dois de seus códigos degeneraram (§2) |
| lifecycle sequencial de VRAM | `application/lifecycle.py` | funcional, pico 4,57 GiB de 8 |
| proveniência e fingerprint encadeado | `domain/references.py`, `application/cache.py` | funcional |

Essa é uma base sólida. O problema não é ausência de estágios; é que **a evidência
produzida não é consumida**.

---

## 2. Limitações reais, medidas

### 2.1 P0 — a evidência alinhada à linguagem é calculada e descartada

`extract_region_evidence` produz `MultiContextResult.visual_embeddings` e
`.language_embeddings`. `run_canonical_pipeline` lê `regions`, `failures`, `metrics` e
`views` — **e ignora os dois campos de embedding**. O que sobrevive é só a string
`artifact_ref` dentro do slot.

Medido no run atual: `model_calls.language_aligned_evidence = 121` por frame
(40 regiões × 3 crops + 1 de cena). Esses 121 vetores são produzidos, contados na
latência e no pico de VRAM, e nunca lidos por ninguém.

Pior: `LanguageAlignedEncoder.encode_text` existe no port, está implementado no adapter
CLIP real e no fake, **e não é chamado em lugar nenhum do código de produção**
(`grep encode_text src/` só encontra a definição e testes). O módulo tem um espaço
imagem-texto compartilhado montado e nunca faz uma única comparação nele.

Consequência: não existe hoje **nenhum** canal de evidência independente do VLM. Toda a
semântica do frame vem de uma única fonte, sem segunda opinião.

### 2.2 P0 — `confidence` do VLM degenerou completamente

`docs/known-limitations.md` registrava 117/177 claims com exatamente `0,90`. No run
atual a degeneração é **total**:

| frame | claims de label primário | valores distintos de `confidence` |
| --- | ---: | ---: |
| `corridor-02-000` | 40 | **1** (`0,90`) |
| `corridor-02-002` | 38 | **1** (`0,90`) |
| `corridor-02-008` | 16 | **1** (`0,90`) |
| `corridor-02-012` | 33 | **1** (`0,90`) |
| `corridor-02-017` | 38 | **1** (`0,90`) |

`diagnostics.json` já marca isso: `semantic_confidence.degenerate = true`,
`stddev = 0.0`, em todos os frames. Qualquer regra de decisão baseada em
`confidence < threshold` é, hoje, uma função constante.

### 2.3 P0 — `contradiction_support` e `contradictory_claims` são ruído estrutural

`claims_compete` trata dois claims `LABEL` de valores diferentes como mutuamente
exclusivos. Como o prompt pede uma lista `alternatives` e o modelo quase sempre devolve
exatamente uma, **toda região com alternativa vira uma contradição**:

- `contradictory_claims`: **163 de 165 regiões** (98,8%);
- `contradiction_support`: média **1,000** em 4 dos 5 frames (0,947 no restante).

Isso é um erro conceitual, não um erro de threshold. Uma alternativa é *incerteza
declarada pelo mesmo produtor na mesma resposta* — é o oposto de duas fontes
independentes discordando. Confundir as duas coisas destrói justamente o sinal que
deveria dirigir o refinamento: `select_refinement_targets` seleciona por
`code == "contradictory_claims"`, ou seja, **selecionaria 163 das 165 regiões**.

### 2.4 P0 — `RegionKind` erra em três de cada quatro regiões

Distribuição no run atual: `thing: 140`, `stuff: 25`. Mas 84 regiões são rotuladas
`wall`/`plain wall`, 41 são `ceiling`/`ceiling tiles`/`ceiling panel`, 10 são
`floor`/`flooring`/`wooden floor`/`carpet`.

Contando regiões cujo label ou category é inequivocamente uma superfície contínua e cujo
`kind` **não** é `stuff`:

| frame | regiões | kind deveria ser `stuff` e não é |
| --- | ---: | ---: |
| `corridor-02-000` | 40 | 30 |
| `corridor-02-002` | 38 | 31 |
| `corridor-02-008` | 16 | 2 |
| `corridor-02-012` | 33 | 30 |
| `corridor-02-017` | 38 | 28 |
| **total** | **165** | **121 (73,3%)** |

O audit reporta apenas **61** (37%), porque `_INHERENTLY_STUFF_CATEGORIES` cobre seis
categorias e o modelo escreve `indoor ceiling`, `ceilings`, `building interior`. A
decisão de manter a lista curta foi correta — ampliá-la seria mascarar o erro — mas o
resultado é que **metade do erro medível é invisível ao audit**.

Falta a peça que a decisão original previa: uma regra determinística que produza
`SUPPORT` ou `CONTRADICTION` sobre a saída do produtor **sem sobrescrevê-la**, e uma
interpretação reconciliada explícita ao lado.

### 2.5 P1 — superfície contínua vira dezenas de regiões, e nada reconcilia

Confirmado e agora quantificado por adjacência. Agrupando regiões do mesmo frame que
compartilham o label primário **e** se tocam (gap ≤ 5 px ou máscaras sobrepostas):

| frame | regiões | grupos adjacentes de mesmo label | regiões cobertas |
| --- | ---: | ---: | ---: |
| `corridor-02-000` | 40 | 7 | 32 |
| `corridor-02-002` | 38 | 6 | 21 |
| `corridor-02-008` | 16 | 1 | 2 |
| `corridor-02-012` | 33 | 7 | 27 |
| `corridor-02-017` | 38 | 6 | 26 |
| **total** | **165** | **27** | **108 (65%)** |

Maiores grupos: `wall` com 13 membros adjacentes em `-000`, `wall` com 10 em `-017`,
`ceiling` com 8 em `-000` e `-017`. E isso é um **piso**: `wall` e `plain wall` são
contados separadamente, então a fragmentação real é maior.

### 2.6 P1 — labels equivalentes contados como conceitos distintos

25 labels primários distintos nos 5 frames, dos quais pelo menos estes pares descrevem o
mesmo conceito: `wall`/`plain wall`, `ceiling`/`ceiling tiles`/`ceiling panel`,
`floor`/`flooring`/`wooden floor`, `row of crops`/`rows of crops`/`field of crops`,
`purple flower`/`pink flowers`, `plain surface`/`flat surface`.

`distinct_labels` continua sendo, portanto, uma métrica contaminada — como
`known-limitations.md` já registrava. O que falta é o outro lado: **um conceito
canônico ao lado do label cru**, não no lugar dele.

### 2.7 P1 — relações são geométricas e só

165 regiões produziram **726 relações** nos 5 frames, todas `geometric_2d`, distribuídas
em `near` (432), `overlaps` (185) e `contains` (109). Zero relações semânticas.

Duas consequências. Primeira: `near` domina (60%) e é O(n²) — `corridor-02-000` sozinho
tem 124 relações `near` para 40 regiões. Segunda: o downstream recebe
"região A encosta na região B" e nada mais. `part_of`, `attached_to`, `supported_by`,
`covers` — tudo que um scene graph precisa — não existe.

### 2.8 P2 — refinamento e reconciliação não estão no caminho canônico

`application/refinement.py` existe, mas:

- não é chamado por `run_canonical_pipeline` (é "extensão pós-pipeline");
- decide por `confidence < 0.5` (constante, §2.2) e `area < 16 px` (`min_mask_area` do
  perfil real é 500, então **nunca dispara**) e por `contradictory_claims` (98,8%,
  §2.3) — as três regras são degeneradas;
- **repete a mesma chamada com a mesma evidência**: `refine_observation` chama
  `interpret_regions` com as mesmas `views`, o mesmo prompt e temperatura 0. É
  literalmente pedir de novo esperando outra resposta.

Reconciliação intra-frame não existe em nenhuma forma.

### 2.9 P2 — `visual_support` não mede suporte visual

`derive_support_inputs` preenche `SupportInputs.visual_support` com
`slot.support_ratio` — a **cobertura de amostragem densa** da máscara. O contract
documenta o campo como "quanta evidência visual independente sustenta a claim". São
coisas diferentes: `support_ratio` vale 1,0 para qualquer região bem amostrada,
independentemente do label estar certo ou errado. No run atual, `visual_support = 1.0`
em praticamente toda claim.

### 2.10 Registrado, sem ação nesta rodada

- `MaskGenerationPipeline` não aceita `point_grids`, então o SAM continua propondo
  máscaras sobre o rig que serão descartadas (custo, não erro) — já em
  `model-backends.md`;
- cast magenta do sensor produz labels literais de cor (`purple cloud`,
  `pink textured surface`) — confundidor real para avaliação futura;
- discovery não é bitwise determinística (jitter de poucos pixels).

---

## 3. O que as sondas mediram

Duas hipóteses precisavam de medição antes de qualquer desenho. Ambas foram testadas com
`benchmarks/evidence_signal_probe.py`, sobre os mesmos frames e as mesmas máscaras do run
`20260908T131207Z`, reconstruindo as views com o próprio `build_region_views` do módulo.

### 3.1 Hipótese A — o alinhamento CLIP arbitra entre as hipóteses do produtor?

Formulada errado, a resposta é decepcionante. Como **classificador open-vocabulary** sobre
o vocabulário do frame, o CLIP acerta o primário do VLM em 6/40 (`masked_subject`) a
13/40 (`tight_crop`) das regiões de `corridor-02-000`, com similaridades numa faixa
estreita (0,13–0,27, mediana 0,19) e margem top1–top2 mediana de **0,005**.

Formulada certo — **árbitro par-a-par entre a hipótese primária e as alternativas que o
próprio produtor registrou** — o sinal aparece (92 regiões com alternativa, 3 frames):

| slot de evidência | concorda com o primary | indistinguível (\|Δ\| < 0,01) | discorda |
| --- | ---: | ---: | ---: |
| `masked_subject` | 50/92 (54%) | 25/92 (27%) | 17/92 (18%) |
| `tight_crop` | 55/92 (60%) | 22/92 (24%) | 15/92 (16%) |
| `contextual_crop` | **59/92 (64%)** | 13/92 (14%) | 20/92 (22%) |

Três leituras importam:

1. **o sinal varia.** Ao contrário de `confidence` (1 valor distinto em 165) e de
   `contradiction_support` (1,000 em quase tudo), aqui há três desfechos com massa real.
   É o primeiro sinal não degenerado que o módulo pode produzir;
2. **o slot certo depende do consumidor.** `masked_subject` é a melhor evidência para o
   VLM (é o sujeito isolado) e a **pior** para o CLIP: 27% de indistinguível, porque o
   CLIP nunca viu recortes sobre fundo cinza chapado no treino. `contextual_crop` é o
   mais decisivo para o CLIP. Um slot único para os dois consumidores seria errado;
3. **a margem é pequena** (mediana 0,019–0,027). Isso proíbe tratar o cosseno como
   confiança e **exige** um desfecho explícito de "indistinguível". Sem ele, 14–27% das
   regiões receberiam um veredito inventado.

### 3.2 Hipótese B — a coerência DINOv2 identifica a mesma superfície?

Sobre todos os pares de regiões de um frame, o cosseno DINOv2 separa mal:
balanced accuracy de 0,55 a 0,66 entre "mesmo label" e "label diferente".

Restringindo aos pares **adjacentes** (617 pares nos 5 frames):

| pares adjacentes | n | p10 | mediana | p90 |
| --- | ---: | ---: | ---: | ---: |
| mesmo label | 131 | 0,407 | **0,781** | 0,957 |
| label diferente | 486 | 0,268 | 0,535 | 0,908 |

E o melhor threshold possível:

| t | recall (mesmo ≥ t) | rejeição (diferente < t) | balanced |
| ---: | ---: | ---: | ---: |
| 0,50 | 0,794 | 0,467 | 0,630 |
| **0,60** | 0,756 | 0,564 | **0,660** |
| 0,70 | 0,641 | 0,644 | 0,643 |
| 0,80 | 0,435 | 0,753 | 0,594 |

**Resultado negativo, e é o mais importante do desenho:** a coerência DINOv2, sozinha,
não decide se duas regiões vizinhas são a mesma superfície. Uma reconciliação que usasse
`cosine ≥ t` como gate erraria um terço das decisões.

A conclusão de desenho é direta: o agrupamento **não pode ser dirigido por similaridade
de feature**. Ele deve ser proposto por **compatibilidade de conceito** (os labels já
concordam, depois de normalização lexical mínima) **mais contato espacial**, e a coerência
densa entra como **corroboração registrada**, capaz de rebaixar um grupo para
`unresolved` quando é baixa — nunca como o critério que cria o grupo.

---

## 4. Comparação com a literatura

### 4.1 O que a literatura já resolveu e nós não usamos

| trabalho | ideia relevante | nosso estado |
| --- | --- | --- |
| **ConceptGraphs** (arXiv:2309.16650) | associa detecções por uma similaridade **composta**: geométrica (`nnratio`) **somada** à semântica (cosseno CLIP normalizado), com limiar explícito; e consolida múltiplas legendas de um mesmo objeto num rótulo final via LLM, com um desfecho `invalid` declarado | temos os dois canais (geometria e CLIP) e **não combinamos nenhum dos dois**. Não há consolidação de hipóteses nem desfecho `invalid`/abstenção efetivo |
| **ConceptGraphs** (edges) | gera relações **só sobre pares candidatos podados** (IoU de caixas → árvore geradora mínima), e dá ao LLM as legendas e posições dos dois objetos | geramos O(n²) relações geométricas e **zero** semânticas; 124 `near` num frame de 40 regiões |
| **Open3DSG** (arXiv:2402.12259) | prevê relações de **conjunto aberto** entre objetos consultáveis, sem taxonomia fechada, condicionando um LLM aos conceitos dos dois nós | nossas relações são três predicados geométricos fixos |
| **VLMaps** (arXiv:2210.05714), **ConceptFusion** (arXiv:2302.07241) | a evidência 2D por região precisa **chegar ao downstream como vetor**, não como rótulo | produzimos os vetores e os descartamos antes de `PipelineResult` (§2.1) |
| **Open-vocabulary detection confidence** (arXiv:2607.10993) | o cosseno CLIP região-texto é uma **mistura enviesada** de escala e especificidade semântica: regiões grandes pontuam mais alto e termos genéricos pontuam mais baixo, independentemente do conteúdo | é exatamente o erro que estaríamos cometendo se transformássemos o cosseno em `confidence`. Sustenta a decisão de emitir *support signal* com margem, e não score |
| **DINOv3** (arXiv:2508.10104) | *Gram anchoring* mantém o mapa denso limpo em resolução alta, onde o DINOv2 degrada | nosso mapa denso é DINOv2-base a 448 (grade 32x24) |
| **SAM 3** (arXiv:2511.16719) | *promptable concept segmentation*: dado um sintagma nominal ou exemplar, devolve **todas** as instâncias correspondentes | nosso discovery é puramente class-agnostic; não há verificação condicionada a conceito |
| **LoftUp** (arXiv:2504.14032) | upsampler baseado em coordenadas com cross-attention, treinado contra pseudo-GT de alta resolução | temos FeatUp/JBU como candidato e `nearest`/`bilinear` como baselines |
| **Qwen3-VL** (arXiv:2511.21631) | família densa 2B/4B/8B/32B, contexto intercalado longo | usamos Qwen2.5-VL-3B em 4-bit |

### 4.2 O paralelo mais próximo do nosso problema

O nosso problema — muitas máscaras class-agnostic da mesma superfície, cada uma com uma
hipótese independente — é **exatamente** o problema que o ConceptGraphs resolve por
associação multi-view em 3D. Eles podem: têm a nuvem de pontos e várias views.

Nós **não podemos**, e não devemos: identidade entre frames e geometria 3D são de
`sensor-association` e `semantic-fusion`. O que podemos fazer, e que ninguém fez por nós,
é a versão **intra-frame e 2D** dessa associação: agrupar por conceito e contato,
registrar como hipótese, preservar toda a geometria original, e deixar a decisão de
identidade para quem tem 3D.

Isso é uma contribuição de composição, não uma reprodução. É importante que a
documentação e o artigo digam isso nesses termos.

---

## 5. O que os papers fazem e **não** pertence a este módulo

Registrado explicitamente para que nenhuma issue desta rodada escorregue para downstream:

| trabalho | o que ele faz | onde isso pertence no projeto |
| --- | --- | --- |
| ConceptGraphs | associação multi-view, fusão em nuvem de pontos, DBSCAN 3D, grafo 3D | `sensor-association`, `semantic-fusion`, `scene-graph` |
| Open3DSG | predição de grafo a partir de nuvem de pontos, co-embedding 3D↔2D | `point-representation`, `scene-graph` |
| VLMaps / CLIP-Fields / OpenScene / ConceptFusion / Open-Fusion | campo semântico espacial, consulta por linguagem sobre o mapa | `semantic-map`, `semantic-memory`, `query-engine` |
| Hydra / HOV-SG | grafo hierárquico online, níveis de floor/room/object | `scene-graph`, `context-reasoning` |
| Vernata / Sonata / ScaLR / DITR / Concerto | representação de pontos e distilação 2D→3D | `point-representation` |
| SAM 2 / SAM 3 (tracking) | identidade de objeto **entre frames** | fora deste módulo por definição |

Em particular: o SAM 3 é interessante aqui **apenas** pela segmentação condicionada a
conceito dentro de um frame. Seu tracker de vídeo é identidade temporal, que é
explicitamente proibida neste módulo.

---

## 6. Oportunidades, classificadas

| # | oportunidade | prioridade | por quê |
| --- | --- | --- | --- |
| O1 | Suporte de hipótese por alinhamento language-aligned (`clip_alignment`), com margem e desfecho `indistinguishable` | **P0** | é o único canal independente do VLM disponível; já pagamos por ele (§2.1) e ele mede não-degenerado (§3.1) |
| O2 | Corrigir a política de exclusividade: alternativa do mesmo produtor ≠ contradição | **P0** | destrava `contradiction_support` e `contradictory_claims` (§2.3), sem os quais nenhum refinamento é possível |
| O3 | Consistência estrutural `label`↔`RegionKind` como SUPPORT/CONTRADICTION preservando o original | **P0** | 73,3% de erro medido, 37% visível (§2.4) |
| O4 | Refinamento seletivo dirigido por razão explícita, escolhendo **nova** evidência | **P0** | as três regras atuais são degeneradas e a chamada é idêntica (§2.8) |
| O5 | Reconciliação contextual intra-frame com grupos de superfície e conceito canônico | **P0** | 65% das regiões estão em grupos adjacentes de mesmo label (§2.5) |
| O6 | Inferência de relações semânticas sobre pares candidatos podados | **P1** | o downstream não recebe nenhuma relação semântica (§2.7) |
| O7 | Integrar tudo em `run_canonical_pipeline` com ordem determinística e audit final | **P1** | capacidades existentes fora do caminho canônico não existem na prática (§2.8) |
| O8 | Expor os embeddings em `PipelineResult` e persistí-los | **P1** | consequência direta de O1; sem isso o downstream recomputa ou fica sem |
| O9 | Benchmark Qwen2.5-VL-3B vs Qwen3-VL-2B/4B sobre a **mesma** arquitetura | **P2** | os três checkpoints já estão no cache local; custo zero de download |
| O10 | Benchmark DINOv2 vs DINOv3 | **P2** | bloqueado: checkpoints `gated=manual` no HF, exigem aceite de licença pelo usuário |
| O11 | SAM 3 como verificação condicionada a conceito em regiões ambíguas | **P2** | bloqueado pelo mesmo motivo (`facebook/sam3`, `gated=manual`) |
| O12 | LoftUp como candidato de upsampling | **P3** | o gargalo medido na #208 era a resolução de entrada do backbone, não a regra de leitura; ganho esperado menor que O1–O7 |
| O13 | Podar `near` O(n²) | **P3** | é custo de serialização, não erro; e a poda correta depende de O6 |

### 6.1 O que deliberadamente **não** entra

- **tabela `label -> kind`**: mascararia o erro do reasoner no momento em que ele ficou
  mensurável. O3 produz contradição registrada, não correção silenciosa;
- **top-N de regiões / merge geométrico agressivo**: descarta evidência para satisfazer
  uma métrica. `known-limitations.md` já mede por que;
- **cosseno como `confidence`**: proibido pela §3.1 e por arXiv:2607.10993;
- **novos campos no `SceneContext`**: nenhum candidato passa nas cinco perguntas da
  Parte I do briefing (global? observável? não vaza objeto? tem consumidor? expressa
  incerteza?). O contract ambiental de seis campos fica como está.

---

## 6.2 Capacidade, problema, solução, prioridade e arquivos

A tabela operacional da revisão. Ela é o índice entre o que foi medido (§2), o que foi
decidido (§6) e onde a decisão vive no código.

| capacidade atual | problema medido | solução | prio | arquivos |
| --- | --- | --- | --- | --- |
| embeddings alinhados a linguagem por região | 121 vetores por frame calculados e descartados; `encode_text` nunca chamado | sinal de suporte por hipótese, com score, margem e status de quatro valores | **P0** | `domain/semantic_support.py`, `application/hypothesis_support.py`, `config.py` |
| confiança do produtor | 165/165 claims com `0,90` exato; `degenerate = true` em todos os frames | continua preservada como evidência bruta; decisões passam a usar sinal, suporte e abstenção | **P0** | `application/semantic_calibration.py`, `application/refinement.py` |
| contradição entre claims | 163/165 regiões marcadas; `contradiction_support` média 1,000 | só hipóteses **afirmadas** competem; ambiguidade vira sinal medido | **P0** | `domain/claim_exclusivity.py`, `domain/semantics.py` |
| `RegionKind` reportado pelo VLM | 119 de 165 regiões contradizem o próprio conceito; o audit via 61 | veredito determinístico compartilhado, por núcleo nominal, que reporta e nunca reescreve | **P0** | `domain/structural_consistency.py`, `application/quality_audit.py` |
| refinamento seletivo | fora do caminho canônico; três regras degeneradas; repetia a mesma chamada | razões explícitas + escalonamento obrigatório de evidência + histórico append-only | **P0** | `application/refinement.py`, `config.py`, `application/pipeline.py` |
| superfície contínua fragmentada | 113 de 165 regiões em grupos adjacentes de mesmo label; nada as reconciliava | grupos de mesma superfície como hipótese, conceito canônico ao lado do label cru | **P0** | `application/reconciliation.py`, `domain/contextual_entities.py` |
| labels lexicalmente equivalentes | `wall`/`plain wall`, `tree`/`trees` contados como conceitos distintos | canonicalização lexical mínima; label cru preservado; `distinct_canonical_concepts` medido | **P1** | `application/reconciliation.py`, `application/observation_diagnostics.py` |
| relações candidatas | 726 relações, todas geométricas; 432 `near`; zero semânticas | inferência sobre pares priorizados, vocabulário fechado e versionado, `none` de primeira classe | **P1** | `application/semantic_relations.py`, `domain/relations.py`, `domain/region_reasoning.py` |
| composição do pipeline | capacidades prontas fora do caminho canônico não executavam | ordem determinística única, audit final depois de tudo que adiciona claim ou aresta | **P1** | `application/pipeline.py` |
| saída para downstream | `artifact_ref` apontando para nada | embeddings expostos no resultado e persistidos por referência | **P1** | `application/pipeline.py`, `benchmarks/frame_artifacts.py` |
| diagnóstico de frame | nada contava o efeito dos estágios contextuais | `ContextualDiagnostics` no artifact e no manifest | **P1** | `application/observation_diagnostics.py`, `benchmarks/validate_reference_pipeline.py` |
| backend de raciocínio multimodal | selecionado antes de a arquitetura contextual existir | benchmark sobre a arquitetura fixa | **P2** | `benchmarks/`, issue #218 |
| backbone denso | gargalo é a resolução efetiva | benchmark de candidato moderno | **P2** | issue #219 (bloqueado por licença) |
| verificação de conceito | não existe | segmentação condicionada a conceito só em regiões não resolvidas | **P2** | issue #220 (bloqueado por licença) |
| upsampling aprendido | FeatUp integrado, não medido na tarefa | comparar candidato mais recente | **P3** | issue #221 |
| relações `near` O(n²) | 432 de 726 relações | poda depende da inferência semântica estar medida | **P3** | issue própria quando houver medida |

## 7. Risco, custo e dependências por oportunidade

| # | risco principal | mitigação | custo VRAM | custo latência | depende de |
| --- | --- | --- | --- | --- | --- |
| O1 | tratar cosseno como confiança | tipo separado, `score` + `margin` + `status`, nunca `ConfidenceScore` | 0 (CLIP já residente no estágio) | +~0,1 s/frame (só encoding de texto, com cache por frame) | — |
| O2 | esconder contradição real entre produtores | só o par primary/alternative **do mesmo produtor** deixa de competir | 0 | 0 | — |
| O3 | virar taxonomia fechada | conjunto pequeno e explícito de conceitos inequívocos; produz claim reconciliada, não sobrescreve | 0 | 0 | O2 |
| O4 | loop infinito, ou refinar tudo | razões explícitas + escalonamento de evidência + término quando não há evidência nova | 0 | +N chamadas VLM, N = regiões não resolvidas | O1, O2, O3 |
| O5 | inventar identidade | grupo é hipótese, geometria imutável, membros preservados | 0 | +~1 s/frame (CPU) | O1, O2, O3 |
| O6 | custo O(n²) de VLM | orçamento de pares configurável, seleção determinística por prioridade geométrica | 0 | +B chamadas VLM, B = orçamento | O5 |
| O7 | regressão de ordem/determinismo | ordem única e testada; audit final depois de tudo que adiciona claim ou relação | 0 | soma dos acima | O1–O6 |
| O8 | inchar a observação serializada | vetores por referência de artifact, nunca inline | 0 | I/O | O1 |
| O9 | atribuir a troca de modelo a mudança de prompt | mesmo prompt, mesmas views, mesmos pixels, só o checkpoint muda | 2B ≈ 2 GB fp16 / 4B ≈ 3 GB 4-bit | comparável | O7 |
| O10 | licença | precisa do aceite do usuário no HF | ViT-B/16 ≈ 86 M params, mesma classe do DINOv2-base | comparável | aceite de licença |
| O11 | vocabulário fechado entrar pela porta dos fundos | SAM3 só **verifica** conceitos que o VLM já propôs, nunca substitui discovery | 860 M params, cabe sozinho | +1 chamada por região ambígua | aceite de licença, O5 |

### 7.1 Orçamento de VRAM

O lifecycle sequencial já garante um modelo residente por vez, com pico medido de
**4,57 GiB** de 8. Nenhuma oportunidade P0/P1 adiciona um modelo novo: O1 usa o CLIP que
já é carregado, O2/O3/O5 são CPU puro, O4/O6 reusam o VLM já residente. O orçamento não
é a restrição desta rodada — a **latência** é.

### 7.2 Estado do hardware nesta sessão

`nvidia-smi` mostra 6,4 GiB de 8 ocupados por um processo não relacionado
(`DD2.exe`, 5,9 GiB) e GPU a 99%. Qualquer execução real precisa esperar a GPU
liberar; medições de VRAM e latência tomadas nessas condições não seriam comparáveis com
os runs versionados.

---

## 8. Referências novas que precisam entrar no artigo

Apenas as que **influenciaram uma decisão concreta** desta revisão. Cada uma está
registrada em `research-traceability.md` com a ideia usada e a diferença para o paper.

| referência | por que entra |
| --- | --- |
| Gu et al., **ConceptGraphs**, arXiv:2309.16650 | similaridade composta geometria+semântica com limiar explícito; poda de pares candidatos antes da inferência de relação; desfecho `invalid` declarado |
| Koch et al., **Open3DSG**, arXiv:2402.12259 | relações de conjunto aberto condicionadas aos conceitos dos nós, sem taxonomia fechada |
| Soon & Hsieh, **Confidence Scores in Open-Vocabulary Detection Are a Biased Mixture of Scale and Semantics**, arXiv:2607.10993 | evidência publicada de que o cosseno CLIP região-texto é enviesado por escala e por especificidade do termo; sustenta emitir *support signal* com margem em vez de confiança |
| Siméoni et al., **DINOv3**, arXiv:2508.10104 | candidato avaliado para features densas (Gram anchoring em alta resolução) |
| Carion et al., **SAM 3**, arXiv:2511.16719 | candidato avaliado para verificação condicionada a conceito |
| Bai et al., **Qwen3-VL**, arXiv:2511.21631 | candidato avaliado para raciocínio multimodal |
| Huang et al., **LoftUp**, arXiv:2504.14032 | candidato avaliado para upsampling de features |

Os quatro trabalhos-base do projeto (VLMaps, CLIP-Fields, o survey de Igelbrink et al. e
Vernata) continuam sustentando a motivação e a fronteira modular, e já estão registrados.

---

## 9. Conclusão da auditoria

A hipótese com que esta revisão começou — *o maior ganho restante está na composição da
evidência, não em mais um ajuste de backend* — é **sustentada pela medição**:

- o módulo produz 121 vetores alinhados à linguagem por frame e lê **zero**;
- os dois sinais que hoje dirigem decisão (`confidence`, `contradiction_support`) têm,
  respectivamente, **1 valor distinto em 165** e **média 1,000**;
- o único sinal independente testado (§3.1) tem três desfechos com massa real e já está
  pago em VRAM;
- 65% das regiões estão em grupos adjacentes de mesmo label que nada reconcilia;
- 73,3% das regiões carregam um `RegionKind` que contradiz o próprio label.

Trocar DINOv2 por DINOv3 ou Qwen2.5 por Qwen3 não move nenhum desses cinco números. A
ordem de trabalho é, portanto, **arquitetura primeiro, backend depois** — e a comparação
de backends só é interpretável depois que a arquitetura estiver fixa, porque só então
"mesma arquitetura, checkpoint diferente" é uma frase verdadeira.

O que esta auditoria **não** autoriza: nenhuma afirmação de superioridade semântica.
Sem `#210`/`#211`, tudo que se pode medir depois da implementação é estrutural — número
de contradições reais, cobertura de abstenção, grupos propostos, relações semânticas
válidas, custo. Acurácia continua bloqueada, e dizer o contrário seria transformar a
saída do próprio pipeline em ground truth.
