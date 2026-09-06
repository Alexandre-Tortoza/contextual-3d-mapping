# Rastreabilidade de pesquisa

Este documento conecta **ideias de pesquisa, decisões do módulo, implementação concreta e
evidência experimental**. Ele existe para evitar duas confusões:

1. tratar uma escolha de engenharia do projeto como se fosse reprodução direta de um artigo;
2. citar um artigo por uma capacidade que o módulo não implementa.

A documentação de arquitetura e as issues permanecem agnósticas a papers quando descrevem
o que deve ser implementado. Esta página é o local explícito para registrar os paralelos
científicos e as diferenças.

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

## Referências principais usadas nesta etapa

| Referência | Contribuição relevante para este projeto |
| --- | --- |
| Huang et al., **Visual Language Maps for Robot Navigation**, arXiv:2210.05714v4 | features visual-language densas associadas à geometria e consulta open-vocabulary espacial |
| Shafiullah et al., **CLIP-Fields: Weakly Supervised Semantic Fields for Robotic Memory**, RSS 2023 | memória espacial 3D aberta a consultas semânticas, integração multi-view e uso de supervisão de modelos web-scale |
| Igelbrink et al., **Online Knowledge Integration for 3D Semantic Mapping: A Survey**, arXiv:2411.18147v1 | decomposição de semantic mapping, integração de VLFMs/LMMs, scene graphs e conhecimento prévio |
| Lemke et al., **Vernata: Self-Supervised Learning of LiDAR Point Representations**, arXiv:2608.06919v1 | representação LiDAR self-supervised, robustez a densidade e distilação cross-modal com features 2D de alta resolução |

## Mapa de influência no `visual-perception`

| Capacidade do módulo | Base científica relevante | Implementação atual | Relação com a referência |
| --- | --- | --- | --- |
| Region discovery | segmentação class-agnostic/promptable moderna | `ports/region_discovery.py`, adapter SAM | usamos a família de segmentação para obter geometria 2D; não reproduzimos uma pipeline específica dos papers principais |
| Dense features | VLMaps, Vernata e literatura de VFMs densos | `ports/feature_extraction.py`, DINOv2, `application/pooling.py` | preservamos estrutura espacial densa para posterior associação e pooling; a fusão 3D fica downstream |
| Language-aligned embeddings | VLMaps, CLIP-Fields | `ports/language_embedding.py`, CLIP | mantemos representação consultável por linguagem, mas ainda no domínio visual 2D |
| Multimodal reasoning | survey sobre VLFMs/LMMs e integração de conhecimento | `ports/multimodal_reasoning.py`, Qwen-VL | usamos VLM para claims estruturados; não delegamos ao VLM a construção do mapa 3D |
| Claims auditáveis | necessidade de lidar com semântica incerta e múltiplas fontes | `domain/semantics.py`, `quality_audit.py` | decisão própria de engenharia para preservar evidência e incerteza |
| High-resolution pooling | motivação de preservar semântica fina em features densas | `application/pooling.py` | decisão própria, alinhada ao problema de granularidade; não é reprodução de Vernata |
| Fusão multi-fonte 2D | motivação geral de integração de múltiplas evidências | `application/fusion.py` | fusão local do módulo; não equivale à fusão multi-view 3D de VLMaps/CLIP-Fields |

## Visual Language Maps

### O que o trabalho demonstra

VLMaps constrói uma representação espacial ao calcular embeddings visual-language por
pixel e associá-los a uma reconstrução 3D. Quando múltiplas observações projetam para a
mesma posição do mapa, as features são agregadas entre views. A representação resultante
pode ser indexada por linguagem natural para localizar landmarks e referências espaciais.

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

| Capability | Seleção atual |
| --- | --- |
| Region discovery | SAM ViT-H |
| Dense features | DINOv2-base |
| Language embedding | CLIP ViT-L/14 |
| Multimodal reasoning | Qwen2.5-VL-3B-Instruct 4-bit |

A metodologia e os valores estão em [model-backends.md](model-backends.md).

Isso não significa que os modelos sejam ótimos de forma universal. Significa apenas que
são a **configuração de referência atual sob o hardware, dataset e proxies usados no
benchmark**.

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
| dense spatial features | VLMaps, Vernata | `feature_extraction.py` + `pooling.py` | permanece 2D neste módulo | benchmark de dense features | implementado |
| open-vocabulary embedding | VLMaps, CLIP-Fields | `language_embedding.py` | região 2D, sem memória 3D | benchmark de language embedding | implementado |
| spatial semantic memory | CLIP-Fields | módulos downstream | não pertence ao `visual-perception` | futura avaliação 3D | fora do escopo deste módulo |
| cross-modal 2D→LiDAR distillation | Vernata | futuro `point-representation`/associação | não implementado aqui | futura avaliação LiDAR | futuro |
| online prior knowledge integration | survey | `scene-graph`/`context-reasoning` | separado da percepção 2D | futura avaliação contextual | downstream |

## Questões em aberto

- calibrar thresholds de refinamento seletivo contra incerteza de backends reais;
- definir protocolo quantitativo específico para qualidade de regiões pequenas;
- medir estabilidade semântica entre frames antes da fusão 3D;
- avaliar se features de resolução superior melhoram a associação 2D→3D em LiDAR
  esparso/distante;
- estabelecer baselines executáveis para comparação histórica com `image-context`;
- definir métricas end-to-end que conectem qualidade de percepção à qualidade final do
  mapa contextual.

Scripts locais são evidência auxiliar. Uma afirmação de pesquisa deve apontar para
resultados reproduzíveis e para a métrica que realmente mede a hipótese em questão.
