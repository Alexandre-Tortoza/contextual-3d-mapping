# Rastreabilidade de pesquisa

Este documento conecta **ideias de pesquisa, decisões do módulo, implementação concreta e
evidência experimental**. Ele existe para evitar duas confusões:

1. tratar uma escolha de engenharia do projeto como se fosse reprodução direta de um artigo;
2. citar um artigo por uma capacidade que o módulo não implementa.

A documentação de arquitetura e as issues permanecem agnósticas a papers quando descrevem
o que deve ser implementado. Esta página é o local explícito para registrar os paralelos
científicos, as diferenças e as referências que devem aparecer no artigo.

## Como ler esta página

```text
referência de pesquisa
        |
        v
ideia relevante
        |
        v
decisão do projeto
        |
        v
arquivo / contract / estágio
        |
        v
benchmark ou avaliação
```

Uma referência sustenta uma motivação ou família de técnica. Ela não prova que nossa
implementação tenha a mesma qualidade, comportamento ou resultados do trabalho citado.

## Referências que passam a compor a base bibliográfica

Os links abaixo apontam para a publicação ou preprint usado como referência de pesquisa.
Quando existir projeto oficial relevante, ele é indicado separadamente.

### Mapeamento visual-language, memória espacial e semantic mapping

| Referência | Pesquisa | Contribuição relevante para este projeto |
| --- | --- | --- |
| Huang et al., **Visual Language Maps for Robot Navigation** | [arXiv:2210.05714](https://arxiv.org/abs/2210.05714) | features visual-language densas associadas à geometria, agregação multi-view e consulta open-vocabulary espacial |
| Shafiullah et al., **CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory** | [arXiv:2210.05663](https://arxiv.org/abs/2210.05663), [projeto](https://clip-fields.github.io/) | memória espacial 3D aberta a consultas semânticas, integração multi-view e supervisão por modelos web-scale |
| Igelbrink et al., **Online Knowledge Integration for 3D Semantic Mapping: A Survey** | [arXiv:2411.18147](https://arxiv.org/abs/2411.18147) | decomposição de semantic mapping, VLFMs/LMMs, scene graphs e integração de conhecimento prévio |
| Lemke et al., **Vernata: Self-Supervised Learning of LiDAR Point Representations** | [arXiv:2608.06919](https://arxiv.org/abs/2608.06919), [código](https://github.com/rai-opensource/vernata) | representação LiDAR self-supervised, robustez a densidade e distilação cross-modal com features 2D de alta resolução |

### Segmentação e region discovery

| Referência | Pesquisa | Relação com o módulo |
| --- | --- | --- |
| Kirillov et al., **Segment Anything** | [arXiv:2304.02643](https://arxiv.org/abs/2304.02643) | base científica direta da família usada pelo `RegionDiscoverer`; o backend de referência atual é SAM ViT-H |
| Ravi et al., **SAM 2: Segment Anything in Images and Videos** | [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) | alternativa moderna de segmentação promptable avaliada na seleção de backend; não é o backend atual |
| Zhao et al., **Fast Segment Anything** | [arXiv:2306.12156](https://arxiv.org/abs/2306.12156) | baseline eficiente comparado na seleção de region discovery; relevante para discutir trade-off qualidade/latência |
| Zhang et al., **Faster Segment Anything: Towards Lightweight SAM for Mobile Applications** | [arXiv:2306.14289](https://arxiv.org/abs/2306.14289) | referência para alternativas compactas de segmentação; não é implementação atual do módulo |

### Dense visual features e open-vocabulary 2D

| Referência | Pesquisa | Relação com o módulo |
| --- | --- | --- |
| Oquab et al., **DINOv2: Learning Robust Visual Features without Supervision** | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | base direta do backend de dense features e da preservação de estrutura espacial antes do pooling |
| Radford et al., **Learning Transferable Visual Models From Natural Language Supervision** | [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) | base do espaço imagem-texto usado pelo backend de language-aligned embeddings |
| Li et al., **Language-driven Semantic Segmentation (LSeg)** | [arXiv:2201.03546](https://arxiv.org/abs/2201.03546) | referência importante para embeddings densos por pixel alinhados à linguagem; também é parte do caminho metodológico de VLMaps |
| Zhou et al., **Detecting Twenty-thousand Classes using Image-level Supervision (Detic)** | [arXiv:2201.02605](https://arxiv.org/abs/2201.02605) | referência de detecção open-vocabulary usada por CLIP-Fields e relevante como alternativa de aquisição semântica por região |

### Multimodal reasoning

| Referência | Pesquisa | Relação com o módulo |
| --- | --- | --- |
| Bai et al., **Qwen2.5-VL Technical Report** | [arXiv:2502.13923](https://arxiv.org/abs/2502.13923) | base do backend de raciocínio multimodal atualmente selecionado para contexto global e interpretação estruturada de regiões |
| Bai et al., **Qwen3-VL Technical Report** | [arXiv:2511.21631](https://arxiv.org/abs/2511.21631), [código](https://github.com/QwenLM/Qwen3-VL) | família densa 2B/4B/8B/32B avaliada como candidata sobre a arquitetura contextual fixa (#218); ainda **não** é o backend selecionado |

### Composição de evidência contextual e grafos abertos

Este grupo é o que sustenta a arquitetura contextual introduzida pelas #214/#204/#205/#206.
Ele é distinto dos grupos acima: aqui não se trata de qual modelo usar, e sim de **como
compor a evidência que vários modelos produzem**.

| Referência | Pesquisa | Contribuição relevante |
| --- | --- | --- |
| Gu et al., **ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning** | [arXiv:2309.16650](https://arxiv.org/abs/2309.16650), [projeto](https://concept-graphs.github.io/) | similaridade composta geometria+semântica com limiar explícito; poda de pares candidatos **antes** de consultar o modelo de linguagem; consolidação de múltiplas descrições em um rótulo, com um desfecho `invalid` declarado |
| Koch et al., **Open3DSG: Open-Vocabulary 3D Scene Graphs from Point Clouds with Queryable Objects and Open-Set Relationships** | [arXiv:2402.12259](https://arxiv.org/abs/2402.12259), [projeto](https://kochsebastian.com/open3dsg) | relações de conjunto aberto condicionadas aos conceitos dos nós, sem taxonomia fechada de predicados |
| Soon & Hsieh, **Confidence Scores in Open-Vocabulary Detection Are a Biased Mixture of Scale and Semantics** | [arXiv:2607.10993](https://arxiv.org/abs/2607.10993) | evidência publicada de que o cosseno região-texto é enviesado por escala da região e por especificidade do termo, e portanto não é probabilidade de acerto |

### Candidatos modernos avaliados

| Referência | Pesquisa | Estado |
| --- | --- | --- |
| Siméoni et al., **DINOv3** | [arXiv:2508.10104](https://arxiv.org/abs/2508.10104), [código](https://github.com/facebookresearch/dinov3) | candidato a dense features; *Gram anchoring* mantém o mapa denso limpo em alta resolução. **Bloqueado** por licença gated no Hub (#219) |
| Carion et al., **SAM 3: Segment Anything with Concepts** | [arXiv:2511.16719](https://arxiv.org/abs/2511.16719), [código](https://github.com/facebookresearch/sam3) | candidato a verificação condicionada a conceito de regiões não resolvidas. **Bloqueado** por licença gated no Hub (#220) |
| Huang et al., **LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models** | [arXiv:2504.14032](https://arxiv.org/abs/2504.14032), [código](https://github.com/andrehuang/loftup) | candidato a upsampling aprendido, alternativo ao FeatUp já integrado. **Não avaliado** nesta rodada (#221) |

### Open-vocabulary 3D mapping e associação 2D→3D

| Referência | Pesquisa | Relação com a arquitetura global |
| --- | --- | --- |
| Peng et al., **OpenScene: 3D Scene Understanding with Open Vocabularies** | [arXiv:2211.15654](https://arxiv.org/abs/2211.15654) | distilação/associação de features abertas à linguagem em pontos 3D; referência direta para a ponte entre percepção 2D e representação 3D |
| Jatavallabhula et al., **ConceptFusion: Open-set Multimodal 3D Mapping** | [arXiv:2302.07241](https://arxiv.org/abs/2302.07241), [projeto](https://concept-fusion.github.io/) | fusão de features multimodais pixel-aligned em mapas 3D e consulta open-set |
| Yamazaki et al., **Open-Fusion: Real-time Open-Vocabulary 3D Mapping and Queryable Scene Representation** | [arXiv:2310.03923](https://arxiv.org/abs/2310.03923), [projeto](https://uark-aicv.github.io/OpenFusion/) | integração online de features region-based com geometria TSDF e representação 3D consultável |

### Representações 3D, distilação 2D→3D e point features

| Referência | Pesquisa | Relação com a arquitetura global |
| --- | --- | --- |
| Wu et al., **Sonata: Self-Supervised Learning of Reliable Point Representations** | [arXiv:2503.16429](https://arxiv.org/abs/2503.16429) | base arquitetural de Vernata e referência para representação self-supervised de pontos sem `geometric shortcut` |
| Puy et al., **Three Pillars improving Vision Foundation Model Distillation for Lidar (ScaLR)** | [arXiv:2310.17504](https://arxiv.org/abs/2310.17504) | mostra a importância conjunta do teacher 2D, backbone 3D e diversidade dos dados na distilação para LiDAR |
| Abou Zeid et al., **DINO in the Room: Leveraging 2D Foundation Models for 3D Segmentation (DITR)** | [arXiv:2503.18944](https://arxiv.org/abs/2503.18944) | projeção/injeção de features 2D em modelos 3D e distilação de VFMs quando imagens não estão disponíveis na inferência |
| Zhang et al., **Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations** | [arXiv:2510.23607](https://arxiv.org/abs/2510.23607) | aprendizado conjunto 2D-3D e representação espacial auto-supervisionada, relevante para evolução de `point-representation` |

### Scene graphs e representação contextual hierárquica

| Referência | Pesquisa | Relação com a arquitetura global |
| --- | --- | --- |
| Hughes et al., **Hydra: A Real-time Spatial Perception System for 3D Scene Graph Construction and Optimization** | [arXiv:2201.13360](https://arxiv.org/abs/2201.13360), [código](https://github.com/MIT-SPARK/Hydra) | construção online de 3D scene graphs hierárquicos e otimização incremental |
| Werby et al., **Hierarchical Open-Vocabulary 3D Scene Graphs for Language-Grounded Robot Navigation (HOV-SG)** | [arXiv:2403.17846](https://arxiv.org/abs/2403.17846), [projeto](https://hovsg.github.io/) | scene graph 3D hierárquico open-vocabulary, com níveis de floor, room e object, diretamente relevante aos módulos `semantic-map` e `scene-graph` |

## Como estas referências entram no artigo

Nem todas devem ser citadas no mesmo parágrafo. A separação recomendada é:

| Parte do artigo | Referências principais |
| --- | --- |
| Percepção visual e region discovery | Segment Anything, SAM 2, FastSAM, DINOv2 |
| Semântica open-vocabulary 2D | CLIP, LSeg, Detic, Qwen2.5-VL |
| Composição de evidência contextual 2D | ConceptGraphs, Open3DSG, Confidence Scores in Open-Vocabulary Detection |
| Candidatos modernos avaliados e não adotados | DINOv3, SAM 3, Qwen3-VL, LoftUp |
| Visual-language mapping e memória espacial | VLMaps, CLIP-Fields |
| Open-vocabulary 3D mapping | OpenScene, ConceptFusion, Open-Fusion |
| 2D→3D distillation e point representations | Sonata, ScaLR, DITR, Concerto, Vernata |
| Semantic mapping e integração de conhecimento | Online Knowledge Integration for 3D Semantic Mapping |
| Scene graph e representação contextual | Hydra, HOV-SG |

Uma referência deve aparecer no texto quando a ideia correspondente for usada para
motivar, comparar ou justificar uma escolha. Não é necessário citar todos esses trabalhos
em toda descrição do módulo.

## Mapa de influência no `visual-perception`

| Capacidade do módulo | Base científica relevante | Implementação atual | Relação com a referência |
| --- | --- | --- | --- |
| Region discovery | Segment Anything, SAM 2, FastSAM | `ports/region_discovery.py`, adapter SAM | usamos segmentação class-agnostic para obter geometria 2D; SAM ViT-H é a referência atual escolhida por benchmark |
| Dense features | DINOv2, VLMaps, Vernata | `ports/feature_extraction.py`, DINOv2, `application/pooling.py` | preservamos estrutura espacial densa para posterior associação e pooling; a fusão 3D fica downstream |
| Language-aligned embeddings | CLIP, LSeg, VLMaps, CLIP-Fields | `ports/language_embedding.py`, CLIP | mantemos representação consultável por linguagem, mas ainda no domínio visual 2D |
| Multimodal reasoning | Qwen2.5-VL e survey sobre VLFMs/LMMs | `ports/multimodal_reasoning.py`, Qwen2.5-VL | usamos VLM para claims estruturados; não delegamos ao VLM a construção do mapa 3D |
| Claims auditáveis | necessidade de lidar com semântica incerta e múltiplas fontes | `domain/semantics.py`, `quality_audit.py` | decisão própria de engenharia para preservar evidência e incerteza |
| High-resolution pooling | DINOv2, Vernata e literatura de dense features | `application/pooling.py` | decisão própria, alinhada ao problema de granularidade; não é reprodução direta de Vernata |
| Fusão multi-fonte 2D | VLMaps, CLIP-Fields, ConceptFusion | `application/fusion.py` | fusão local do módulo; não equivale à fusão multi-view 3D dessas referências |
| Saída preparada para 2D→3D | OpenScene, ConceptFusion, Open-Fusion, DITR | `VisualObservation` + fronteira de `sensor-association` | preservamos geometria, embeddings e proveniência para que a associação 3D ocorra downstream |

## Visual Language Maps

### O que o trabalho demonstra

VLMaps constrói uma representação espacial ao calcular embeddings visual-language por
pixel e associá-los a uma reconstrução 3D. Quando múltiplas observações projetam para a
mesma posição do mapa, as features são agregadas entre views. A representação resultante
pode ser indexada por linguagem natural para localizar landmarks e referências espaciais.

Pesquisa: [arXiv:2210.05714](https://arxiv.org/abs/2210.05714).

O ponto relevante para `visual-perception` é a necessidade de produzir evidência visual
que preserve **localização espacial fina e alinhamento semântico**, em vez de reduzir a
imagem inteira a um único texto ou label.

### O que implementamos de forma relacionada

```text
imagem RGB
    -> dense feature map
    -> mask-aware pooling
    -> embedding visual por região
    -> language-aligned embedding
    -> VisualObservation
```

Arquivos:

- [`ports/feature_extraction.py`](../src/visual_perception/ports/feature_extraction.py);
- [`application/pooling.py`](../src/visual_perception/application/pooling.py);
- [`ports/language_embedding.py`](../src/visual_perception/ports/language_embedding.py);
- [`domain/regions.py`](../src/visual_perception/domain/regions.py).

### O que não implementamos aqui

`visual-perception` não projeta as features em uma reconstrução 3D e não agrega views no
mesmo grid espacial. Essas responsabilidades pertencem a `sensor-association`,
`semantic-fusion` e módulos de mapa.

Portanto, citar VLMaps como motivação para features spatially grounded é correto. Afirmar
que `visual-perception` implementa VLMaps não é.

## CLIP-Fields

### O que o trabalho demonstra

CLIP-Fields aprende uma função espacial que associa coordenadas 3D a representações
semânticas e visuais. O trabalho mostra que impor estrutura 3D e consistência entre views
pode melhorar previsões em relação ao modelo 2D usado como supervisão, além de permitir
consultas semânticas e visuais sobre a cena.

Pesquisa: [arXiv:2210.05663](https://arxiv.org/abs/2210.05663), [projeto](https://clip-fields.github.io/).

### Influência relevante

A principal lição para a arquitetura global é que a evidência 2D não deve ser o fim do
processo. Ela deve carregar representação suficiente para que módulos downstream possam
integrar observações espacialmente e recuperar informação por linguagem.

No `visual-perception`, isso aparece como:

- embeddings preservados por referência;
- `ObservedRegion` com geometria 2D explícita;
- separação entre evidência visual e entidade espacial;
- saída preparada para `sensor-association`.

### Diferença importante

CLIP-Fields é uma memória espacial implícita treinada por cena. O módulo atual não treina
um neural field e não mantém memória 3D. Essa capacidade pertence às etapas posteriores
da arquitetura do projeto.

## Survey de Online Knowledge Integration

### O que a referência organiza

O survey descreve semantic mapping como combinação de três problemas gerais:

1. geometric mapping;
2. aquisição de informação semântica a partir de sensores;
3. integração de conhecimento prévio e reasoning.

Também discute como VLFMs, LMMs e scene graphs aproximam representações geométricas,
semânticas e simbólicas.

Pesquisa: [arXiv:2411.18147](https://arxiv.org/abs/2411.18147).

### Relação com nossa fronteira modular

`visual-perception` ocupa principalmente a etapa de **aquisição de informação semântica
visual**.

```text
geometric mapping
    -> fora de visual-perception

semantic information from RGB
    -> visual-perception

2D/3D association and fusion
    -> downstream

prior knowledge / scene graph / reasoning
    -> downstream
```

Isso sustenta a decisão de não transformar o módulo em um sistema monolítico de semantic
mapping.

### Consequência arquitetural

O VLM pode produzir claims, mas esses claims continuam evidência. Conhecimento prévio,
contexto espacial 3D e reasoning global não devem corrigir silenciosamente a observação
2D dentro deste módulo sem uma fronteira explícita e auditável.

## Vernata

### O que o trabalho demonstra

Vernata adapta self-distillation para LiDAR outdoor e adiciona:

- sparse view augmentation para robustez a variação de densidade;
- memory bank para estabilizar treinamento com compute limitado;
- cross-modal distillation a partir de features 2D de alta resolução.

Na ablação reportada pelo trabalho, a distilação cross-modal produz o maior ganho
individual entre as extensões avaliadas. O artigo também mostra vantagem da associação de
features de maior resolução em pontos LiDAR mais distantes.

Pesquisa: [arXiv:2608.06919](https://arxiv.org/abs/2608.06919), [código](https://github.com/rai-opensource/vernata).

### Relação com `visual-perception`

O módulo atual **não implementa cross-modal distillation** e não deve alegar isso.

A relevância está em produzir uma fonte 2D de boa resolução e semântica que possa servir
como evidência para associação ou distilação em módulos 3D futuros.

Isso reforça a importância de:

- não reduzir dense features cedo demais;
- preservar geometria de masks;
- manter embeddings/proveniência recuperáveis;
- avaliar regiões pequenas e detalhes em alta resolução.

Arquivos relacionados:

- [`ports/feature_extraction.py`](../src/visual_perception/ports/feature_extraction.py);
- [`domain/feature_map.py`](../src/visual_perception/domain/feature_map.py);
- [`application/pooling.py`](../src/visual_perception/application/pooling.py).

A implementação de distilação 2D para LiDAR pertence principalmente a
`point-representation`/`sensor-association`, não a este módulo.

## Referências adicionais e fronteira correta de citação

### Segment Anything, SAM 2 e variantes eficientes

Essas referências sustentam a família metodológica de **region discovery**, não a
semântica final das regiões.

O pipeline usa propostas geométricas class-agnostic e adiciona semântica posteriormente.
Portanto:

- SAM pode ser citado como base direta do backend atual;
- SAM 2 e FastSAM devem ser citados quando os resultados do benchmark de seleção forem
  apresentados;
- Faster Segment Anything/MobileSAM pode ser citado ao discutir alternativas compactas,
  sem afirmar que foi implementado ou benchmarkado no pipeline atual.

### DINOv2

DINOv2 é a referência direta para o backend de dense feature extraction atualmente
selecionado. Deve ser citado sempre que o artigo descrever a origem das features visuais
self-supervised usadas antes do pooling por região.

Pesquisa: [arXiv:2304.07193](https://arxiv.org/abs/2304.07193).

### FeatUp

FeatUp fundamenta o adapter opcional de elevação aprendida JBU da #208. O projeto fixa a
revisão do repositório oficial e preserva o checkpoint do upsampler na provenance. Essa
referência sustenta o método de elevação; não sustenta, sem o benchmark local, ganho na
tarefa semântica do `corridor-02`.

Pesquisa: [ICLR 2024](https://openreview.net/forum?id=GkJiNn2QDF).
Código oficial: [mhamilton723/FeatUp](https://github.com/mhamilton723/FeatUp).

### CLIP, LSeg e Detic

CLIP fundamenta o espaço compartilhado imagem-texto utilizado pelo módulo.

LSeg é relevante porque mostra um caminho explícito para features densas por pixel
alinhadas à linguagem e também participa da construção original de VLMaps.

Detic é relevante como método open-vocabulary por região e como parte da supervisão usada
por CLIP-Fields.

Essas referências ajudam a posicionar o módulo entre abordagens **pixel-level**,
**region-level** e **image-text embedding**, mesmo quando o backend concreto escolhido é
diferente.

### Qwen2.5-VL

Qwen2.5-VL deve ser citado porque é o modelo concreto usado pelo backend de
`MultimodalReasoner` da configuração real de referência.

Pesquisa: [arXiv:2502.13923](https://arxiv.org/abs/2502.13923).

A citação sustenta a escolha do modelo. Ela não transforma claims gerados pelo VLM em
verdade de ground truth.

### OpenScene, ConceptFusion e Open-Fusion

Esses trabalhos são referências diretas para o próximo limite arquitetural após
`visual-perception`: associar evidência aberta à linguagem à geometria 3D e manter uma
representação consultável.

Eles devem ser usados para contextualizar:

```text
visual-perception
    -> evidência 2D estruturada
    -> sensor-association
    -> semantic-fusion
    -> semantic-map
```

Não devem ser descritos como implementados dentro de `visual-perception`.

### Sonata, ScaLR, DITR e Concerto

Esse grupo fundamenta a linha de **representações 3D auto-supervisionadas** e
**transferência/distilação 2D→3D**.

A relação com o projeto é principalmente futura e downstream:

- `point-representation` aprende ou mantém features 3D;
- `sensor-association` fornece correspondências 2D↔3D;
- features de `visual-perception` podem atuar como evidência ou teacher;
- Vernata é a referência mais diretamente alinhada ao caso LiDAR outdoor do projeto.

### Hydra e HOV-SG

Esses trabalhos sustentam a evolução de uma representação geométrica/semântica para uma
representação contextual hierárquica e relacional.

Eles pertencem principalmente à justificativa dos módulos `semantic-map`, `scene-graph`
e `context-reasoning`, não ao funcionamento interno de `visual-perception`.

## Arquitetura contextual: paper -> princípio -> decisão -> diferença

Esta seção existe porque a rodada da #214/#204/#205/#206 é a primeira em que uma decisão
de **composição** — e não de escolha de modelo — foi informada por literatura. Cada linha
diz exatamente qual ideia foi usada e onde a nossa implementação diverge.

### ConceptGraphs

**O que o trabalho faz.** Constrói um scene graph 3D open-vocabulary a partir de RGB-D:
máscaras class-agnostic por frame, um embedding CLIP por máscara, associação entre frames
por uma similaridade **composta** (`ϕ = ϕ_semântica + ϕ_geométrica`, com limiar explícito),
consolidação de várias legendas de um objeto em um rótulo final via LLM, e arestas geradas
apenas sobre pares podados por IoU de caixas seguido de árvore geradora mínima.

**Princípio usado, e só ele.** Três ideias, todas de composição:

1. **similaridade composta com limiar explícito** — geometria e semântica entram na mesma
   decisão, e nenhuma delas decide sozinha;
2. **poda de pares antes de consultar o modelo** — perguntar sobre todos os pares é O(n²);
3. **um desfecho declarado de "nada aqui"** — eles têm `invalid` para detecções; nós temos
   `none` para relações e `indistinguishable` para hipóteses.

**Nossa decisão.** `application/reconciliation.py` propõe grupos por **conceito compatível
mais contato espacial**, com a coerência densa entrando como corroboração registrada;
`application/semantic_relations.py` seleciona pares por prioridade geométrica dentro de um
orçamento antes de consultar o VLM.

**Diferença para o paper, e ela é grande.**

| ConceptGraphs | nós |
| --- | --- |
| associa **entre frames**, em 3D, sobre nuvem de pontos | agrupa **dentro de um frame**, em 2D, sobre máscaras |
| a associação produz identidade de objeto | o agrupamento produz uma **hipótese**, e todos os membros continuam existindo |
| a similaridade semântica **decide** a associação | a nossa coerência densa **corrobora**; quem propõe é conceito + contato |
| consolida legendas em um rótulo final | acrescenta um conceito canônico **ao lado** do label cru |

A quarta linha não é preferência: é medição. Entre pares adjacentes dos nossos frames, o
cosseno denso separa mesmo-label de label-diferente com acurácia balanceada de **0,660** no
melhor limiar possível. Um gate por similaridade erraria um terço das decisões, e por isso
ele não é o gate.

**O que não podemos afirmar.** Que implementamos ConceptGraphs. Não fazemos associação
multi-view, não fundimos em nuvem de pontos, não construímos grafo 3D. Essas capacidades
pertencem a `sensor-association`, `semantic-fusion` e `scene-graph`.

**Onde citar.** Ao descrever a reconciliação intra-frame e a poda de pares de relação.

### Open3DSG

**O que o trabalho faz.** Prevê grafos de cena 3D com classes **e relações** de conjunto
aberto, co-embedding features de um backbone 3D com o espaço de um modelo 2D aberto, e
condicionando um LLM aos conceitos consultados dos nós.

**Princípio usado.** Condicionar a inferência de relação aos **conceitos dos dois nós**, em
vez de classificar a aresta a partir de geometria pura.

**Nossa decisão.** `RegionRelationRequest` carrega o conceito reconciliado e a natureza de
cada lado, junto da view do par e da medida geométrica já calculada.

**Diferença.** O nosso vocabulário de predicados é **fechado e versionado**, e o deles é
aberto. A razão é a fronteira do módulo: seis dos predicados abertos que um LLM produziria
(`above`, `behind`, distância, alcançabilidade) exigem profundidade, e afirmá-los a partir
de um único frame seria inventar geometria. Um grafo downstream também precisa de tipos de
aresta estáveis para significar alguma coisa. Os **conceitos** continuam abertos; o tipo de
aresta, não.

**Onde citar.** Ao descrever a inferência de relações semânticas e a escolha de manter o
vocabulário de predicados fechado.

### Confidence Scores in Open-Vocabulary Detection

**O que o trabalho mostra.** O score `cos(v, t)` de detectores open-vocabulary é uma
mistura enviesada: regiões grandes pontuam mais alto que pequenas para a mesma consulta, e
termos genéricos pontuam mais baixo que específicos para o mesmo conteúdo visual.

**Princípio usado.** Um cosseno região-texto **não é** probabilidade de acerto, e o viés
tem direção conhecida.

**Nossa decisão.** `HypothesisSupportSignal` guarda `score` e `margin` como medidas
nomeadas, com um `status` de quatro valores, e **nunca** como `ConfidenceScore`. A decisão
é tomada sobre a **margem** entre hipóteses da mesma região — que compartilham a escala,
cancelando o primeiro viés — e um piso explícito produz `indistinguishable` em vez de um
vencedor inventado.

**Diferença.** Eles propõem uma correção de temperatura em tempo de teste para recuperar
recall de objetos pequenos. Nós não corrigimos o score: preferimos não convertê-lo em
confiança nenhuma enquanto não houver ground truth para calibrar (#210/#211).

**Onde citar.** Ao justificar por que o suporte de hipótese emite sinal e não confiança.

### O que esta rodada mediu por conta própria

Duas medições não vêm de paper nenhum e sustentam decisões próprias:

| medição | valor | decisão que ela sustenta |
| --- | --- | --- |
| CLIP como árbitro par-a-par, por slot | concorda 54–64%, indistinguível 14–27%, discorda 16–22% | o sinal existe e não é degenerado; o slot melhor para o VLM (`masked_subject`) é o pior para o CLIP |
| coerência densa entre pares adjacentes | balanced accuracy 0,660 no melhor limiar | a similaridade densa **corrobora** grupos, e não os cria |

Ambas são reproduzíveis por `benchmarks/evidence_signal_probe.py`.

## Decisões próprias de engenharia

As decisões abaixo não devem ser atribuídas diretamente aos papers acima sem evidência
mais específica.

### Claims em vez de label vencedor

[`domain/semantics.py`](../src/visual_perception/domain/semantics.py) preserva múltiplas
hipóteses semânticas, evidência, proveniência e confiança opcional.

Objetivo:

- manter discordâncias visíveis;
- permitir auditoria;
- evitar transformar ausência de score em certeza artificial;
- permitir decisões downstream com contexto adicional.

### Política de confiança ausente

`confidence=None` significa que o produtor não forneceu score. A política de seleção vive
em `most_confident_claim` e impede claims não pontuados de vencer claims pontuados.

Essa é uma decisão de domínio deste projeto.

### Fingerprints encadeados

[`application/cache.py`](../src/visual_perception/application/cache.py) invalida um
estágio e seus dependentes quando configuração upstream muda. Essa é uma decisão de
reprodutibilidade e engenharia, não uma contribuição reproduzida dos papers.

### Relações 2D candidatas

[`application/relation_generation.py`](../src/visual_perception/application/relation_generation.py)
mantém relações geométricas e inferidas como candidatas. Elas não são promovidas a
relações 3D sem validação downstream.

### Sinal de suporte com três desfechos

O terceiro desfecho — `indistinguishable` — é decisão própria, motivada por medição: 14 a
27% das regiões dos frames de referência têm margem abaixo do ruído do próprio sinal.
Colapsá-lo em "sem suporte" inventaria um veredito que a medida não sustenta.

### Coerência estrutural como veredito, não como correção

[`domain/structural_consistency.py`](../src/visual_perception/domain/structural_consistency.py)
emite `SUPPORTS`/`CONTRADICTS`/`UNDETERMINED` sobre a natureza declarada de uma região, e
**não** reescreve a saída do produtor. Casar pelo núcleo nominal do composto, e não por
qualquer palavra dele, também é decisão própria: é o que impede `ceiling light fixture` de
ser classificado como superfície contínua.

### Papel `RECONCILED`

Um terceiro papel de hipótese, ao lado de `PRIMARY` e `ALTERNATIVE`, para que a
interpretação reconciliada exista **ao lado** da original em vez de no lugar dela. Nenhum
dos trabalhos revisados preserva as duas: o ConceptGraphs consolida em um rótulo final.

### High-resolution mask-aware pooling

[`application/pooling.py`](../src/visual_perception/application/pooling.py) possui caminho
para regiões menores que a resolução nominal do feature map. A motivação é preservar
informação de regiões pequenas; a implementação é própria do projeto.

### Fusão multi-fonte

[`application/fusion.py`](../src/visual_perception/application/fusion.py) combina evidência
sem apagar claims discordantes. Essa fusão é interna ao domínio 2D e não deve ser descrita
como fusão 3D multi-view.

## Estado experimental atual

A escolha de checkpoints **não está mais pendente**. O benchmark #174 selecionou a
configuração real de referência para a RTX 3060 8GB:

| Capability | Seleção atual | Referência de pesquisa |
| --- | --- | --- |
| Region discovery | SAM ViT-H | [Segment Anything](https://arxiv.org/abs/2304.02643) |
| Dense features | DINOv2-base | [DINOv2](https://arxiv.org/abs/2304.07193) |
| Language embedding | CLIP ViT-L/14 | [CLIP](https://arxiv.org/abs/2103.00020) |
| Multimodal reasoning | Qwen2.5-VL-3B-Instruct 4-bit | [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) |

A metodologia e os valores estão em [model-backends.md](model-backends.md).

Isso não significa que os modelos sejam ótimos de forma universal. Significa apenas que
são a **configuração de referência atual sob o hardware, dataset e proxies usados no
benchmark**.

Quando os resultados comparativos de region discovery forem usados no artigo, também
cite as referências dos candidatos efetivamente comparados, especialmente SAM 2 e
FastSAM.

## Evidência necessária para mudar uma decisão

Uma mudança de referência deve registrar:

- hipótese;
- dataset e subset;
- baseline;
- candidato;
- checkpoint;
- configuração;
- hardware;
- métrica quantitativa apropriada;
- latência e memória quando relevantes;
- resultados brutos;
- análise de falhas;
- revisão do código.

Para uma mudança inspirada por artigo, registre também:

- qual ideia específica foi adotada;
- qual parte foi modificada;
- por que a adaptação é necessária;
- o que o artigo mede;
- o que nosso experimento mede;
- onde as comparações deixam de ser equivalentes.

## Matriz de rastreabilidade

Use este formato para novas mudanças relevantes:

| Ideia | Referência/motivação | Implementação | Diferença nossa | Evidência | Status |
| --- | --- | --- | --- | --- | --- |
| class-agnostic region discovery | SAM, SAM 2, FastSAM | `region_discovery.py` + adapter SAM | backend atual é SAM ViT-H | benchmark #174 | implementado |
| dense spatial features | DINOv2, VLMaps, Vernata | `feature_extraction.py` + `pooling.py` | permanece 2D neste módulo | benchmark de dense features | implementado |
| open-vocabulary embedding | CLIP, LSeg, VLMaps, CLIP-Fields | `language_embedding.py` | região 2D, sem memória 3D | benchmark de language embedding | implementado |
| multimodal structured claims | Qwen2.5-VL | `multimodal_reasoning.py` + parsers em `application/` | VLM produz evidência, não ground truth | validação de saída estruturada | implementado |
| spatial semantic memory | CLIP-Fields | módulos downstream | não pertence ao `visual-perception` | futura avaliação 3D | fora do escopo deste módulo |
| open-vocabulary 3D map | OpenScene, ConceptFusion, Open-Fusion | `sensor-association` + `semantic-fusion` + mapa | não implementado aqui | futura avaliação 3D | downstream |
| cross-modal 2D→LiDAR distillation | ScaLR, DITR, Concerto, Vernata | futuro `point-representation`/associação | não implementado aqui | futura avaliação LiDAR | futuro |
| self-supervised point representation | Sonata, Vernata | `point-representation` | módulo ainda separado da percepção visual | linear probing/3D downstream | futuro |
| online prior knowledge integration | survey | `scene-graph`/`context-reasoning` | separado da percepção 2D | futura avaliação contextual | downstream |
| hierarchical open-vocabulary scene graph | Hydra, HOV-SG | `semantic-map` + `scene-graph` | arquitetura própria e modular | futura avaliação contextual | downstream |
| suporte de hipótese por alinhamento | Confidence Scores in Open-Vocabulary Detection | `application/hypothesis_support.py` | emitimos sinal com margem, nunca confiança; três desfechos em vez de dois | sonda de sinal de evidência | implementado |
| reconciliação intra-frame | ConceptGraphs | `application/reconciliation.py` | intra-frame e 2D; grupo é hipótese, membros preservados; coerência corrobora, não decide | sonda de coerência entre pares adjacentes | implementado |
| relações semânticas condicionadas ao conceito | Open3DSG, ConceptGraphs | `application/semantic_relations.py` | vocabulário de predicados fechado e versionado; poda de pares por prioridade geométrica | validação real de três frames | implementado |
| coerência conceito/natureza | decisão própria | `domain/structural_consistency.py` | veredito, nunca correção; casamento por núcleo nominal | contagem no run de referência | implementado |

## Questões em aberto

- calibrar thresholds de refinamento seletivo contra incerteza de backends reais;
- definir protocolo quantitativo específico para qualidade de regiões pequenas;
- medir estabilidade semântica entre frames antes da fusão 3D;
- avaliar se features de resolução superior melhoram a associação 2D→3D em LiDAR
  esparso/distante;
- comparar explicitamente abordagens de associação/fusão com OpenScene, ConceptFusion e
  Open-Fusion quando os módulos 3D estiverem executáveis;
- avaliar distilação 2D→3D contra ScaLR, DITR, Concerto e Vernata quando
  `point-representation` estiver pronto;
- estabelecer baselines executáveis para comparação histórica com `image-context`;
- definir métricas end-to-end que conectem qualidade de percepção à qualidade final do
  mapa contextual.

Scripts locais são evidência auxiliar. Uma afirmação de pesquisa deve apontar para
resultados reproduzíveis e para a métrica que realmente mede a hipótese em questão.
