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
| `samples/20260907T014526Z/` | `50ed564` | estado atual, `prompt_version v5` |

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
verificação estrutural depois do parsing, não apenas de uma frase no prompt.

Uma correção plausível é comparar o label produzido contra os valores das
`scene_claims` daquela mesma observação e recusar, ou pelo menos marcar, a coincidência —
mas isso ainda não foi implementado nem medido, e não deve ser tratado como resolvido.

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

Cuidado ao ler os overlays: o número ao lado do label é a `geometric_confidence` da
região, não a confiança do claim.

---

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
