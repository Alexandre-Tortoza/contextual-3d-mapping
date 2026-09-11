# Densidade temporal no corridor-02 — leitura dos resultados

Companheiro de `report.md`, que é gerado e contém as tabelas. Este arquivo é
escrito à mão e interpreta aqueles números. Nenhuma alteração de percepção,
mapa ou consumer foi feita para produzi-los.

## O que foi executado

| | A (referência) | B (candidato) |
|---|---|---|
| run | `20260910T183839Z` | `20260910T211622Z` |
| janela | `corridor-02-176s-30s-window.json` | `corridor-02-176s-15s-1fps-window.json` |
| trecho | t0+176,3 s, 30 s | t0+176,3 s, **15 s** |
| amostragem | 16 keyframes a 2 s | 16 frames a **1 FPS** |
| mapa | `corridor-02-176s-30s-context.json` | `corridor-02-176s-15s-1fps-context.json` |

`config_fingerprint` idêntico nos dois (`130a3cf0…`), mesma revisão `0131e25`,
mesmo prompt v8, mesmo checkpoint, mesmo `sequence-masks`, mesmo
`scene_context_mode`. O slice geométrico e a odometria do FAST-LIO foram
reaproveitados: `play_offset_s`, `play_duration_s` e `pose_anchor_ns` são
idênticos por construção, então a geometria não é uma variável.

**Design aninhado.** Os alvos pares de B caem exatamente sobre os 16 keyframes
de A no mesmo trecho: 8 frames são compartilhados (PNGs byte-idênticos por
SHA-256) e 8 são novos, intercalados a cada 1 s.

**Controle de não-determinismo: zero.** Os 8 frames compartilhados produziram
resultados idênticos nos dois runs — mesma contagem de regiões, mesmos
conceitos, sem uma única divergência. Com `temperature=0.0` o pipeline é
determinístico, então **toda** diferença entre A e B é atribuível à amostragem.

## Respostas às perguntas do experimento

### 1. O pallet verdadeiro é detectado consistentemente conforme a câmera se aproxima?

**Não. Ele é publicado em exatamente 1 frame de 16.**

O vocabulário de pallet (`pallet`, `plank`, `wood`, `crate`, `timber`, `box`,
`debris`) aparece somente em t=1,98 s em toda a sequência de 15 s. Em t=1,00 s
**o pallet está claramente visível** encostado na parede direita — verificado na
imagem — e nenhuma região o nomeia, nem como observação publicada nem como
contexto estrutural. Em t=3,00 s ele já saiu do quadro.

Dobrar a densidade não recuperou o pallet no frame vizinho onde ele é visível.
Isso descarta a hipótese de amostragem para este objeto: **a perda é
instabilidade de detecção, não janela temporal**.

### 2. O falso pallet reaparece? É persistente ou alucinação pontual?

**Alucinação pontual.** A região de 210 456 px (box `(8,4)-(640,337)`, quase o
frame inteiro) rotulada `wooden pallet` ocorre em t=1,98 s e em nenhum outro
frame. Ela não é reforçada pela densidade — o que é boa notícia — mas também
não é refutada, porque o mapa não tem fusão temporal para desempatar: no
artifact contextual ela continua sendo uma região publicada de peso igual ao
pallet verdadeiro, que é 10× menor.

### 3. `broken tile` permanece reconhecido? As duas regiões são a mesma entidade?

**Sim para as duas perguntas, com uma ressalva.** `broken tile` é publicado em
t=1,98 s (2 regiões) e t=5,00 s (1 região), com `broken concrete floor` e
`broken floor tile` em t=4,02 s e `broken floor` em t=5,00 s — um agrupamento
de dano contínuo entre t=2 e t=5.

As duas regiões de t=1,98 s **já são agrupadas** numa única
`entity_hypotheses` (`status: supported`, coerência de feature 0,97, ambos os
membros rotulados `broken tile`). A reconciliação intra-frame faz o trabalho
certo. A ressalva é que **o artifact do mapa publica as duas regiões
separadamente** e ignora o agrupamento — a duplicata é reconhecida na
observação e perdida na composição.

### 4. A porta no fim do corredor passa a ser reconhecida em frames posteriores?

**Não.** Portas aparecem em t=0, t=1, t=6,02 e t=7,97 e **em nenhum outro
frame**. As de t=6 e t=8 são as portas laterais no cruzamento, não a porta do
fundo. Em t=5,00 s há duas portas vermelhas escuras claramente visíveis nas
laterais e nenhuma foi publicada. Não há progressão de reconhecimento com a
aproximação; há aparecimento e desaparecimento sem relação com o tamanho
aparente.

### 5. Aparecem evidências que os 16 keyframes não capturaram?

**Sim, três, e uma delas importa.**

- **`peeling paint`** em t=5,00 s, em 2 regiões (84 762 px e 3 286 px). É uma
  classe de dano **completamente ausente** de A. Verificado na imagem: a parede
  direita tem reboco descascando de forma inequívoca. **As duas regiões são
  exibidas como `wall`** — ver a seção "O defeito que não é de amostragem".
- **`black glove`** em t=15,02 s (1 677 px) — objeto pequeno, nenhum keyframe o
  captou.
- **`ceiling vent`** em t=1,00 s e t=15,02 s — conceito novo em relação a A.

O resto do que os frames novos acrescentaram é reforço do que os keyframes já
tinham (luminárias em t=13, dano de piso em t=5) ou descrição não-comprometida.

### 6. Quantos frames consecutivos ficam sem evidência contextual?

**No máximo 1** em B (t=3,00 s), contra 0 em A no mesmo trecho e 2 em A no
trecho completo de 30 s. Cobertura parcial: os 3 frames vazios de A estão na
segunda metade da janela, que esta sonda não cobriu.

O frame vazio de B é o achado mais informativo da sequência. Em t=3,00 s o piso
está severamente danificado — ladrilhos soltos, subpiso exposto, detritos por
todo o quadro — e as 39 regiões observadas foram rotuladas `wall` (23),
`ceiling` (8), `floor` (4), `surface` (3) e `ceiling panel` (1). **Nenhuma como
dano.** A supressão estrutural fez exatamente o que devia; o que falhou foi o
raciocínio de região, que descreveu piso destruído como piso.

### 7. Os objetos ficam semanticamente estáveis?

**Não, mas a instabilidade é entre frames, não dentro deles.** Dentro de um
frame a reconciliação resolve as hipóteses concorrentes de forma consistente
(14 regiões publicadas em B carregam primárias divergentes e todas foram
reconciliadas). Entre frames, o mesmo conteúdo físico alterna entre
`ceiling light fixture` / `ceiling fixture` / `ceiling` / `ceiling vent`, e
entre `broken tile` / `broken floor tile` / `broken concrete floor` /
`broken floor` — quatro nomes para o mesmo dano de piso em 4 segundos.

### 8. As máscaras ficam mais coerentes conforme o viewpoint melhora?

**Não há evidência disso.** As regiões de maior área continuam sendo as menos
específicas em qualquer densidade: a maior região publicada de B é a `door` de
270 080 px em t=6,02 s (66% do frame útil), e a segunda é o falso pallet de
210 456 px. A área da região publicada não correlaciona com aproximação; ela
correlaciona com o segmentador ter fundido a cena inteira num blob.

## O defeito que não é de amostragem

**8 das 52 regiões publicadas em B (15%) exibem um substantivo estrutural puro
como rótulo** — e em **todos** os casos a região foi publicada porque uma
hipótese irmã carregava evidência real:

| frame | exibe | hipótese que causou a publicação |
|---|---|---|
| 04354 (t=5,00) | `wall` | **`peeling paint`** |
| 04354 (t=5,00) | `wall` | **`peeling paint`** |
| 04379 (t=6,02) | `wall` | `smooth surface` |
| 04402 (t=6,99) | `wall` | `white wall` |
| 04402 (t=6,99) | `wall` | `pink wall` |
| 04475 (t=10,01) | `wall` | **`ceiling fan`** |
| 04498 (t=10,99) | `ceiling` | **`ceiling light fixture`** |
| 04570 (t=14,00) | `ceiling` | **`ceiling light fixture`** |

Em A o mesmo padrão ocorre em 4 de 36 (11%).

O mecanismo está documentado no próprio código:
`domain/contextual_evidence.py:asserted_identity_claims` julga **todas** as
hipóteses afirmadas de uma região e publica se qualquer uma não for superfície
estrutural genérica — decisão correta e deliberada. Mas o rótulo exibido vem da
reconciliação intra-frame, que escolhe **uma**. As duas regras podem discordar,
e quando discordam a evidência que justificou a publicação é a que some.

Consequência prática: **`peeling paint` está sendo detectado e jogado fora na
exibição.** Não é um falso positivo de `wall` na supressão; é evidência de dano
real publicada sob o nome errado.

Há uma segunda camada disso no mapa: `corridor-02-context` escolhe a **primeira
claim `primary`**, não a reconciliada, e diverge em 3 das 52 regiões
(`black cylindrical object` vs `black rectangular object`, `blue carpet` vs
`blue textured surface`, `broken floor tiles` vs `broken floor tile`).

Nada disso foi corrigido, conforme o controle experimental.

## A × B

### Recall contextual

| | A (mesmo trecho, 8 kf) | B (16 f @ 1 FPS) |
|---|---|---|
| regiões publicadas | 29 | 52 |
| publicadas por frame | 3,62 | 3,25 |
| conceitos distintos | 16 | 26 |
| frames sem evidência | 0/8 | 1/16 |

Dobrar os frames rendeu +79% de regiões publicadas, mas **menos publicações por
frame**. Os 8 frames intercalados produziram 23 regiões — 21% a menos que os 8
keyframes.

### Precisão aparente

Classificando manualmente as 23 publicações dos 8 frames **novos**, contra as 29
dos 8 keyframes:

| | keyframes (8) | frames novos (8) |
|---|---|---|
| evidência contextual genuína | 24 (83%) | 14 (61%) |
| superfície estrutural sem portador real | 3 (10%) | 4 (17%) |
| descrição não-comprometida | 1 (3%) | 5 (22%) |
| falso positivo declarado | 1 (3%) | 0 |

As descrições não-comprometidas — `black rectangular object` (3×),
`grid pattern`, `blue textured surface` — quintuplicaram. Elas ocupam uma vaga
de observação contextual no mapa sem dizer nada acionável.

### Estabilidade temporal e persistência

Das 26 concepções publicadas em B, **18 aparecem em um único frame**
(persistência trivial de 1,00 sobre span 1). Só 7 conceitos sobrevivem a mais de
um frame, e os dois de maior span têm a menor continuidade:
`ceiling light fixture` (4 frames num span de 14, persistência 0,29) e
`ceiling vent` (2 frames num span de 15, persistência 0,13).

Nenhum conceito tem persistência 1,00 sobre um span maior que 1. **Nada no
trecho é observado de forma contínua**, nem paredes, nem portas, nem dano — o
que é a formulação mais direta do problema: o pipeline não é temporalmente
estável em nenhuma escala testada.

### Mapa

| eixo | A (30 s) | B (15 s) |
|---|---|---|
| observações visuais | 16 | 16 |
| pontos contextuais | 19 912 | **24 371** |
| pontos com RGB | 86 236 | 79 187 |
| pontos geométricos não observados | 50 391 | 57 440 |
| pontos multi-observação | 43 519 | 51 743 |
| suporte corroborado | 3 657 | **6 952** |
| suporte não corroborado | 16 088 | 16 880 |
| suporte fraco | 167 | 539 |
| corroborado / contextual | 18,4% | **28,5%** |
| regiões contextuais no mapa | 36 | 52 |

Este é o único eixo em que a densidade ganha de forma inequívoca. **B cobre
metade da trajetória e ainda assim tem 22% mais pontos contextuais e 90% mais
pontos corroborados.** Por metro de corredor, a amostragem regular
aproximadamente dobra a cobertura contextual corroborada — porque cada ponto é
visto de mais viewpoints e a associação sensorial tem mais votos.

O custo aparece no suporte fraco, que triplicou (167 → 539), e em `wall`
subindo de 2 para 6 regiões no mapa — as tais publicações com rótulo errado.

Não há saturação por observação repetida: os 136 627 pontos geométricos são os
mesmos nos dois mapas, e o crescimento está na fração deles que recebeu
contexto, não em pontos duplicados.

Ressalva de comparação: B cobre os primeiros 15 s da trajetória, A cobre 30 s.
As 4 regiões `window` de A e as 3 frames vazias estão na segunda metade, fora
desta sonda.

## Resposta à pergunta central

> Uma amostragem temporal regular e mais densa permite recuperar evidências
> contextuais relevantes que estavam sendo perdidas pelos keyframes esparsos?

**Marginalmente, e não é onde está o problema.**

Três evidências decisivas, todas livres de ruído porque o pipeline é
determinístico:

1. o pallet está **visível** em t=1,00 s e não é detectado; a densidade dobrada
   não o recuperou;
2. o frame t=3,00 s mostra dano de piso severo e classifica 39 de 39 regiões
   como superfície genérica;
3. em t=5,00 s duas portas vermelhas visíveis não são publicadas, enquanto o
   `peeling paint` da mesma frame é publicado sob o nome `wall`.

O que a densidade **entregou** foi real mas modesto em recall (`peeling paint`,
`black glove`, `ceiling vent`) e substancial em mapa (corroboração de 18,4% para
28,5%). O que ela **não** entregou foi estabilidade: nenhum conceito é observado
continuamente, e os frames adicionais têm precisão 61% contra 83% dos keyframes.

## Recomendação

**E — o problema não está principalmente na amostragem temporal; voltar ao
visual-perception.**

O gargalo medido é o raciocínio de região: ele descreve dano como superfície
genérica, alterna entre quatro nomes para o mesmo dano em quatro segundos, e
deixa de nomear objetos que estão nítidos no quadro. Mais frames dão mais
oportunidades de acertar, mas a taxa de acerto por frame não muda, e a precisão
dos frames adicionais é menor.

Com uma qualificação separada: **adotar C (1 FPS regular) para a composição do
mapa**, não por recall, mas pelo número de corroboração — dobrar a cobertura
contextual corroborada por metro é um ganho medido e barato de obter. As duas
decisões são independentes e devem ser tomadas como tais.

Antes de qualquer mudança de amostragem, três itens deste run têm retorno maior
e custo menor, em ordem:

1. **exibir a hipótese que causou a publicação**, e não a reconciliada. Recupera
   `peeling paint` — uma classe de dano inteira — sem tocar em modelo, prompt ou
   threshold. Alinhar `corridor-02-context` na mesma regra elimina a segunda
   divergência;
2. **propagar o agrupamento de entidade para o mapa**. As duas regiões de
   `broken tile` já são reconhecidas como uma entidade `supported` com coerência
   0,97 e mesmo assim viram dois pontos de evidência independentes;
3. **investigar por que o segmentador publica regiões que cobrem o frame
   inteiro** (270 080 px e 210 456 px). O falso pallet e a `door` gigante são o
   mesmo modo de falha, e nenhuma política de publicação a jusante consegue
   distinguir um blob de cena inteira de um objeto.

`semantic_confidence` continua degenerada em 0,90 nos 16 frames de B — sem
mudança em relação a A, como esperado, já que nada de confiança foi tocado.

## O que ficou fora

- A segunda metade da janela (t+15 s a t+30 s), onde estão os 3 frames vazios de
  A e as 4 regiões `window`. Uma sonda de 15 s foi executada em lugar dos 30 s
  para conter o custo de GPU (75,5 min medidos, 283 s/frame).
- Nenhum tracking ou deduplicação temporal foi implementado. A coluna
  `persistência` do `report.md` é co-ocorrência de conceito, não identidade de
  objeto.
- Nenhuma métrica de acurácia: o conjunto não tem anotação humana revisada, e
  usar a saída do pipeline como ground truth não produziria número interpretável.
