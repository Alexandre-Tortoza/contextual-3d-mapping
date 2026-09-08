# Limitações conhecidas

Este documento registra falhas **observadas em execução real** que continuam abertas.
Ele existe para que uma limitação medida não precise ser redescoberta pelo próximo ciclo,
e para que nenhuma delas seja confundida com um problema já resolvido.

Cada item traz o que foi medido, onde, e qual issue é dona da correção. Uma limitação só
sai daqui quando um run real mostrar que ela deixou de ocorrer — não quando o código que
deveria corrigi-la for escrito.

## Origem das medidas

Todas as contagens abaixo vêm dos runs versionados dos três frames vinculantes
`corridor-02-000`, `corridor-02-008` e `corridor-02-017` (126 regiões canônicas no total):

| run | revisão | o que era |
| --- | --- | --- |
| `samples/20260906T195310Z/` | `ebe211f` | baseline: crop pelo bounding box, cena achatada em string, `prompt_version v2` |
| `samples/20260907T005820Z/` | `0fda5cf` | primeira versão multi-view, `prompt_version v3` |
| `samples/20260907T014526Z/` | `50ed564` | `prompt_version v5`, sem exclusão de área |
| `samples/20260907T115208Z/` | `abb755b` | último run antes da #202: 64 proposals → 60 regiões |
| `samples/20260908T003114Z/` | #202 | estado atual, `prompt_version v6`, áreas declaradas |

## O que a #202 fechou, medido em `corridor-02-002`

Comparação direta entre `samples/20260907T115208Z/` e `samples/20260908T003114Z/`, mesmo
frame, mesmo SHA-256 de entrada:

| campo | antes | depois |
| --- | ---: | ---: |
| `proposal_count` | 64 | 64 |
| `region_count` | 60 | **38** |
| regiões sobre o rig | 19 | **0** |
| regiões fora da lente | 7 | **0** |
| `duplicate_label_hypotheses` | 7 regiões | **0** |
| `scene_echo_label_count` | 2 | **0** |
| `dominant_label` | `plain wall` 18/60 (0,30) | `plain wall` 14/38 (0,37) |
| `distinct_labels` | 14 | 11 |
| `RegionKind` estruturado | inexistente | 38/38 |
| `category` estruturada | inexistente | 37/38 |
| `suitcase` no contexto de cena | sim | **não** |
| `latency_s` | 271 | 181 |

As 24 proposals descartadas se explicam inteiramente: 13 por sobrepor o rig, 8 por caírem
fora do círculo útil da lente, 3 por área relativa mínima. `proposal_count` menos
`merged_proposal_count` fecha exatamente, e `rejected_proposals` diz o motivo de cada uma.

**Sobre `dominant_fraction`:** ela subiu de 0,30 para 0,37, e isso não é uma regressão de
interpretação. O denominador caiu de 60 para 38 regiões, e a contagem absoluta de
`plain wall` caiu de 18 para 14. O que a fração mostra é que as regiões removidas eram
majoritariamente as *outras* — rig e vinheta —, não as paredes. A over-segmentação de
superfície contínua permanece exatamente onde estava, e é a limitação 4 abaixo.

---

## 1. Contexto de cena promovido a label de região

**Status: aberto.** Dona: `#202`.

No run atual, 6 das 50 regiões de `corridor-02-017` saem rotuladas com a cena inteira:
5 como `long, narrow hallway` e 1 como `long narrow hallway`. Em `corridor-02-000`
aparece uma ocorrência de `long hallway`.

Isso é exatamente o que a `#202` proíbe: nenhuma claim de nível de cena deve virar
verdade da região sem evidência local própria. A implementação atual já entrega as claims
de cena estruturadas ao reasoner e o prompt afirma explicitamente que elas descrevem a
cena e "must not be repeated as the label" — e mesmo assim o modelo as repete.

O que isso significa na prática: a instrução textual **não é suficiente** como mecanismo
de garantia. A separação entre propriedade de cena e propriedade de região precisa de uma
verificação estrutural, não apenas de uma frase no prompt — mas, como a `#212` mediu
abaixo, o canal que precisava ser fechado não era o textual.

### O que a `#212` mediu

A `#212` separou os canais por onde a cena pode alcançar a região —
`scene_context_mode` (textual, as claims no prompt) e `region_views` (visual,
`contextual_crop` e `scene_conditioned`) — e passou a contar
`scene_echo_label_count`: regiões cujo label repete uma claim de cena.

Medido em `corridor-02-002`, variando **um fator de cada vez** sobre a mesma revisão:

| ego-mask | `contextual_crop` | claims de cena | proposals→regiões | labels distintos | `dominant_fraction` | `scene_echo` |
| --- | --- | --- | --- | --- | --- | --- |
| aplicada | sim | sim | 47→45 | 1 | **1,00** (`curved wall`) | **45/45** |
| aplicada | não | sim | 47→45 | 3 | 0,96 (`curved wall`) | 43/45 |
| **não** | **sim** | **sim** | 64→60 | 14 | **0,30** (`plain wall`) | 2/60 |
| não | não | sim | 64→60 | 12 | 0,50 (`plain wall`) | 2/60 |
| não | não | não | 64→60 | 18 | 0,53 (`plain wall`) | 0/60 |

**A causa raiz era a própria ego-mask destrutiva.** Tirar o `contextual_crop` moveu o
colapso de 1,00 para 0,96; parar de pintar a faixa preta moveu de 0,96 para 0,30.

As duas hipóteses com que esta mudança começou foram **reprovadas pela medição**, e ambas
os defaults que elas motivaram foram revertidos:

- desligar o canal visual (`contextual_crop`) **piora**: com pixels íntegros, ligado dá
  0,30 e desligado dá 0,50. Ele ajuda a desambiguar quando a entrada não está corrompida;
- desligar o canal textual (`local_first`) zera o eco residual, mas mede levemente pior no
  colapso (0,53 contra 0,50 aqui; 0,52 contra 0,42 em `corridor-02-000`, ambos medidos sem
  `contextual_crop`).

O único default que mudou, portanto, foi parar de destruir os pixels. Na configuração
resultante — pixels íntegros, `contextual_crop` ligado, claims de cena no prompt — os dois
frames vinculantes desta investigação medem:

| frame | antes (`abb755b`) | depois |
| --- | --- | --- |
| `corridor-02-000` | 62→54 regiões, 14 labels, dominante 0,52 | 75→66, **21 labels**, dominante **0,21** |
| `corridor-02-002` | 47→45 regiões, **1 label**, dominante **1,00** | 64→60, **14 labels**, dominante **0,30** |

O aumento de proposals (62→75 e 47→64) é esperado e é o custo aceito: sem a faixa preta, o
chassi do robô volta a gerar regiões.

O mecanismo, agora visível: zerar as linhas ≥ 340 corrompia a **análise de cena**. Com a
faixa preta somada à vinheta do fisheye, `analyze_scene` passava a emitir `curved wall`
como claim de nível de cena — e daí todas as 45 regiões a ecoavam. O vazamento cena→região
que esta limitação descreve era real e massivo, mas quem o disparava era o
pré-processamento, não o canal de contexto.

Com os pixels íntegros, o eco cai para 2/60 mesmo com as claims de cena no prompt. A
garantia estrutural do `local_first` existe, está testada e continua ablatável por
`--scene-context-mode` — ela só não é o default, porque a medição não a sustenta.

A lição de método é a mesma que a #212 codificou como regra: **os pixels de origem são
imutáveis**. A máscara existia para remover uma distração conhecida — o chassi do robô — e
acabou produzindo uma falha sistemática de rotulagem muito pior que a distração que
removia. Ela era invisível porque o overlay era desenhado sobre a imagem já destruída.

**Status: fechada pela #202.** O vazamento textual foi eliminado estruturalmente, e não
por instrução de prompt: `REGION_SCENE_CLAIM_KINDS` passou a conter **apenas** claims
ambientais (`scene_type`, `environment`, `layout`, `lighting`, `visibility`,
`navigability`). Prosa livre e inventário de objetos deixaram de existir no contract de
cena, de modo que não há o que ecoar. Medido no run atual: `scene_echo_label_count` = 0.

A parte que **continua aberta** é outra, e está registrada como limitação 4: 14 das 38
regiões ainda saem como `plain wall`. Isso não é mais eco de cena — é over-segmentação de
uma superfície contínua, e o número não melhora sem reconciliação semântica entre regiões
do mesmo frame.

## 2. Labels não são normalizados

**Status: aberto.** Sem issue dona; candidato natural à `#205` (reconciliação intra-frame).

`long, narrow hallway` e `long narrow hallway` são contados hoje como dois labels
distintos, diferindo apenas por uma vírgula. O mesmo vale para pares como `tree`/`trees`
e `ceiling light`/`ceiling light fixture`, todos presentes no run atual.

A consequência é metodológica e importa para qualquer comparação futura: **contagem de
labels distintos é uma métrica contaminada**. Parte da variação que ela mostra é ruído de
superfície morfológica, não conteúdo semântico novo. Ao comparar runs, a contagem de
não-respostas genéricas (`plain wall`, `plain surface`) é um proxy mais honesto, porque
não depende de normalização.

Normalizar tem um custo real que precisa ser decidido conscientemente: colapsar
`ceiling light` e `ceiling light fixture` descarta uma distinção que pode ser legítima. A
reconciliação da `#205` é o lugar certo para essa decisão, porque ela vê todas as regiões
do frame ao mesmo tempo.

## 3. O backend nunca omite `confidence`

**Status: aberto.** Dona: `#196` para o tratamento, `#210` para a evidência que o
desbloqueia.

O contract representa ausência de score corretamente (`SemanticClaim.confidence=None`), e
o prompt pede explicitamente que o modelo **omita** a chave quando não souber estimá-la.
Medido em `ebe211f`, nos três frames: 177 claims de label, **177 pontuadas, zero
omitidas**. A distribuição concentra-se em poucos valores âncora — 117 das 177 valem
exatamente `0,90`, e 138 ficam em `0,90` ou acima (mínimo `0,05`, máximo `0,98`).

O score emitido pelo Qwen2.5-VL-3B se comporta mais como hábito de formatação do que como
estimativa de incerteza. Tratá-lo como confiança calibrada seria um erro, e nenhuma
decisão downstream deve depender do seu valor bruto.

É para isso que existe a calibração da `#196` — que por sua vez continua bloqueada pela
anotação humana revisada da `#210`. Enquanto esse bloqueio existir, o eixo "incerteza" de
qualquer comparação permanece **inalterado e ruim**, e nenhuma métrica de calibração
(ECE, Brier, taxa de hallucination) pode ser publicada.

**Corrigido na `#212`** o caveat de leitura que acompanhava esta limitação: o overlay
imprimia `label (0.97)`, onde o número era a `geometric_confidence` da máscara e era lido
como certeza semântica. Os dois eixos agora aparecem nomeados — `sem=0.90 geom=0.97`, com
`sem=?` quando o produtor não pontuou — e um teste de regressão impede o formato antigo de
voltar. A limitação em si, o backend nunca omitir `confidence`, **continua aberta**: o que
mudou foi a legibilidade do artifact, não o comportamento do modelo.

---

## 4. Superfície contínua vira muitas regiões

**Status: aberto.** Candidato natural à `#205` (reconciliação intra-frame).

No run atual de `corridor-02-002`, 14 das 38 regiões saem como `plain wall` e 11 como
`wall`. Somadas, 25 das 38 regiões descrevem *a mesma parede*, recortada em pedaços pelo
SAM.

A #202 não atacou isso, e as regras que ela introduziu deliberadamente não conseguem
atacar: elas decidem **validade** (dentro do sensor, fora do rig, tamanho plausível), e a
redundância geométrica continua com `merge_regions`. Um pedaço de parede adjacente a outro
não é inválido nem duplicado — tem IoU baixo com os vizinhos e evidência própria.

Foi medida e recusada uma regra de containment: para uma máscara inteiramente contida em
outra, `IoU == |contida| / |continente|`, de modo que uma regra de containment separada só
dispara quando as áreas diferem muito — isto é, descartando uma **parte** dentro de um
todo, que é evidência legítima que o módulo preserva desde a #158
(`test_contained_part_is_not_removed`).

Resolver isso exige decidir que duas regiões vizinhas são a **mesma entidade**, o que é
uma pergunta semântica e não geométrica. É exatamente a canonicalização proposal→entidade
que a `#205` possui.

**Consequência para comparações:** `region_count` neste frame não vai abaixo de ~38 por
geometria. A meta de menos de 25 regiões não foi atingida, e atingi-la com `top-N` teria
sido descartar evidência para satisfazer uma métrica.

## 5. `RegionKind` é reportado, mas o modelo erra muito

**Status: aberto.** Dona: `#205`/`#203` conforme onde a evidência melhorar.

O contract agora carrega `RegionKind` até a serialização, e o run atual o traz em 38 de 38
regiões. Mas a distribuição é `thing: 34, stuff: 4`, e a auditoria acusa **16** ocorrências
de `region_kind_inconsistent_with_category` — 15 delas categoria `wall` com kind `thing`.

Uma parede é `stuff` em qualquer cena. O modelo está errando, e o audit torna isso
contável em vez de invisível.

O que **não** foi feito, deliberadamente: sobrescrever a saída do modelo com uma tabela
`label -> kind`. Isso mascararia o erro do reasoner exatamente no momento em que ele
passou a ser mensurável. O conjunto de invariantes do audit é minúsculo por isso
(`wall`, `floor`, `flooring`, `ceiling`, `ground`, `sky`) e só sinaliza.

Note que ele não cobre tudo: `indoor ceiling` e `ceilings` aparecem no run e não estão no
conjunto, então as três regiões com `indoor ceiling`/`thing` passam sem warning. Ampliar o
conjunto é fácil e é justamente o que **não** se deve fazer sem antes melhorar a
evidência: o caminho é medir se o modelo acerta mais com views melhores, não estender a
tabela até o número ficar bonito.

## Observações menores, registradas para não se perderem

### O cast de cor da câmera puxa labels literais

Os frames do `corridor-02` têm um cast magenta forte, com vegetação rosa e céu azul — a
assinatura de um sensor sensível a near-infrared. Em `corridor-02-008` isso produz labels
literais de cor como `purple cloud` e `pink textured surface`.

Não é um defeito do pipeline: o modelo descreve os pixels que recebe. Mas é um confundidor
real para qualquer avaliação de acurácia semântica nesse dataset, e precisa constar do
protocolo antes de comparar contra ground truth humano.

### `corridor-02-008` não é um corredor

Apesar do nome do dataset, o frame `corridor-02-008` mostra o robô em campo aberto, com
árvores, céu e uma caixa d'água ao fundo. O `scene_type: field` produzido pelo modelo está
**correto**, e não deve ser tratado como erro de cena em nenhuma análise futura.

### O chassi do robô volta a ser rotulado

**Resolvido pela #202.** O texto abaixo descreve o estado anterior e fica registrado
porque a lição de método continua valendo.

A solução estrutural existe agora: a silhueta do rig é geometria declarada em
`benchmarks/sequence-masks/corridor-02.json`, e a exclusão acontece na filtragem de
proposals, sobre máscaras. Os pixels de origem continuam imutáveis
(`pipeline_input_identical_to_raw` = `true` no run atual), e `--mask-ego-vehicle` foi
removido junto com o caminho destrutivo. Medido: 13 proposals rejeitadas por sobreposição
com o rig, e **0** regiões finais sobre ele.

Só uma das duas barreiras planejadas foi implementada. A exclusão dos pontos de prompt
antes da inferência do SAM **não é possível** com a integração atual: verificado em
`transformers` 5.16.1, `MaskGenerationPipeline._sanitize_parameters` não aceita
`point_grids` — a grade é gerada internamente por `image_processor.generate_crop_boxes`.
O detalhe está em [model-backends.md](model-backends.md). O custo é computação gasta em
propostas que serão descartadas; o gate é atingido pela barreira de rejeição sozinha.

O texto original:

Até a `#212`, o validador zerava a faixa inferior da imagem (linhas ≥ 340) antes do SAM,
para o chassi do robô não virar região. Isso alterava os pixels de entrada, e o overlay era
desenhado sobre a imagem já destruída — nada no artifact revelava a alteração.

Agora os pixels de origem são imutáveis por default, e a máscara do ego-veículo é
persistida como artifact independente (`ego-mask.png`). A consequência esperada é real: o
chassi volta a aparecer rotulado nos runs. É uma piora visível de labels em troca de
entrada fiel, e `diagnostics.json` a torna contável em vez de escondida.

`--mask-ego-vehicle` reproduzia o comportamento antigo quando era preciso comparar com
runs anteriores; nesse caso `pipeline-input.png` era gravado ao lado de `raw.png` e a
diferença entre os dois ficava visível. A flag e o caminho destrutivo foram removidos pela
#202, junto com a solução estrutural descrita acima.

### A descoberta de regiões não é bitwise determinística

Entre dois runs da mesma configuração, `corridor-02-017` apresentou 2 masks diferentes de
50 — **2 pixels de 307.200**, com `region_id`, box e ordem idênticos. É jitter numérico do
SAM sob estado de memória de GPU diferente, não mudança de geometria causada por reasoning.

Consequência para comparações: um digest de geometria bitwise é sensível demais para
decidir se um estágio semântico alterou a geometria. Compare `region_id`, box e ordem, e
trate diferenças de mask da ordem de poucos pixels como ruído do backend.

## Uma lição de método, medida neste módulo

O texto do prompt é uma **variável de primeira ordem** do resultado — nestes frames, maior
que a escolha de views de evidência.

Com os **mesmos pixels** (`--region-view tight_crop` reproduz exatamente o crop do
baseline), as mesmas claims de cena e o mesmo exemplo, só trocando a frase que enquadra a
tarefa, o label `door` em `corridor-02-000` foi de 1/54 para 45/54 e de volta para 3/54.
Pedir para *identificar* o sujeito faz o modelo nomear um objeto discreto — num corredor,
uma porta. Pedir para *descrever o que está visível* mantém a resposta presa aos pixels.

Duas consequências operacionais:

- uma ablation de evidência que não fixe o prompt **não mede o que diz medir**. Use
  `--region-view` para variar a evidência com prompt constante;
- uma mudança de evidência e uma de prompt entregues juntas podem se cancelar. Foi o que
  quase aconteceu na `#203`: um ganho real ficou escondido atrás de uma regressão de
  prompt até a ablation separar os dois.

O detalhe completo está em [model-backends.md](model-backends.md), seção
"Contract de resposta de região".
