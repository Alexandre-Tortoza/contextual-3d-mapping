# Backends de modelos

`visual-perception` separa capacidades de percepção dos modelos concretos. A aplicação depende dos ports; os adapters em `infrastructure/adapters/` escolhem e carregam os backends reais.

## Capacidades substituíveis

| Capacidade | Responsabilidade | Backend atual | Checkpoint |
| --- | --- | --- | --- |
| Region discovery | propor masks e regiões sem exigir classe | SAM ViT-H | `facebook/sam-vit-huge` |
| Dense feature extraction | produzir feature map visual denso | DINOv2-base | `facebook/dinov2-base` |
| Language embedding | alinhar imagem e texto | CLIP ViT-L/14 | `openai/clip-vit-large-patch14` |
| Multimodal reasoning | interpretar cena e regiões | Qwen2.5-VL-3B-Instruct | `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit |

Esses modelos descrevem a configuração `research_quality` e a reference run `20260910T115810Z`.

O adapter de region discovery usa o pipeline `mask-generation` do Transformers e pode acomodar checkpoints SAM/SAM2 compatíveis. Entretanto, a configuração de referência atual seleciona **SAM ViT-H**. Uma documentação de run deve sempre citar o checkpoint realmente utilizado em vez de tratar SAM e SAM2 como equivalentes.

## Onde cada backend entra

```text
RGB
├── RegionDiscoverer
│      -> SAM
│      -> RegionProposal[]
│
├── DenseFeatureExtractor
│      -> DINOv2
│      -> FeatureMap [Hf, Wf, C]
│
├── LanguageAlignedEncoder
│      -> CLIP image encoder
│      -> CLIP text encoder
│
└── MultimodalReasoner
       -> Qwen
       -> SceneContext + SemanticClaim[]
```

Factories e adapters concretos ficam em [`infrastructure/adapters/`](../src/visual_perception/infrastructure/adapters/). Fakes determinísticos ficam em [`infrastructure/fakes/`](../src/visual_perception/infrastructure/fakes/).

## SAM

SAM resolve **segmentação/proposta de regiões**, não identidade semântica.

Entrada:

```text
RGB image
```

Saída conceitual:

```text
mask [H, W]
bounding box
geometric confidence
```

Exemplo:

```text
SAM:
    "estes pixels parecem pertencer à mesma região"

não:
    "esta região é uma porta"
```

A máscara depois é usada para construir views para Qwen/CLIP e para selecionar evidência DINO.

## DINOv2

DINO produz features densas por patch.

Na configuração atual:

```text
frame original: 640 x 480
input long edge: 448
processado: 448 x 336
patch size: 14 x 14
feature grid: 32 x 24
dimension: 768
```

Logo:

```text
FeatureMap.shape = [24, 32, 768]
```

Cada posição espacial possui um vetor 768D. Esses vetores descrevem conteúdo visual aprendido, não labels humanas.

## CLIP

CLIP possui dois encoders compatíveis:

```text
image -> vector[768]
text  -> vector[768]
```

No pipeline, os image embeddings são produzidos para views como `masked_subject`, `tight_crop` e `contextual_crop`.

Depois que Qwen propõe hipóteses como:

```text
wooden panel
wooden door
```

os mesmos textos são codificados pelo CLIP text encoder e comparados com os image embeddings.

```text
CLIP(image crop) dot CLIP("wooden door")
```

Isso produz suporte relativo à hipótese, não uma probabilidade calibrada.

## Qwen

Qwen executa raciocínio multimodal em dois níveis principais:

```text
scene context
    -> interpreta ambiente global

region semantics
    -> interpreta uma região usando views derivadas da máscara
       + contexto ambiental estruturado
```

O Qwen atual não recebe diretamente o vetor DINO como entrada do reasoner. Ele recebe pixels e contexto estruturado.

## Relação entre os quatro

```text
SAM
    define onde olhar

DINO
    descreve visualmente patches e regiões

Qwen
    propõe significado

CLIP
    mede compatibilidade entre a evidência visual e os conceitos textuais
```

Exemplo real resumido:

```text
Qwen primary:     wooden panel
Qwen alternative: wooden door

CLIP masked_subject:
    door > panel

CLIP contextual_crop:
    panel > door
```

A pipeline preserva essa divergência para refinement e calibração.

## Regra arquitetural

O domínio não deve importar bibliotecas específicas de SAM, DINO, CLIP ou Qwen. Dependências pesadas, device, quantização, carregamento de checkpoint e transforms específicos ficam na infraestrutura.

Isso permite:

- testar a pipeline sem GPU;
- trocar modelos em benchmarks sem reescrever o domínio;
- comparar backends por capacidade;
- registrar proveniência do modelo em cada observação;
- manter contracts estáveis quando o backend muda.

## Feature upsampling

Existe uma fronteira de feature upsampling em `infrastructure/adapters/feature_upsampling_backend.py`. Ela altera a forma de consultar a evidência densa, não cria uma nova fonte de labels.

Interpolar ou elevar a resolução não significa que o backbone passou a observar detalhes nativos menores que seu patch sem informação adicional.

## Como documentar uma troca de modelo

Ao mudar um backend, registre no mínimo:

- capacidade substituída;
- nome e checkpoint;
- parâmetros que alteram o comportamento observável;
- resolução, patch size ou shape relevante;
- precisão/quantização e device quando afetarem o resultado;
- impacto medido em benchmark ou artifact versionado;
- configuração usada para reprodução.

Não atualize esta página com um modelo apenas planejado. O backend deve existir no código ou em benchmark explicitamente identificado como candidato.

## Próxima leitura

- [Pipeline detalhada de Visual Perception](./pipeline.md)
- [02. Region Discovery](../../../docs/02-region-discovery.md)
- [04. Dense Feature Extraction](../../../docs/04-dense-features.md)
- [07. Language-Aligned Evidence](../../../docs/07-language-aligned-evidence.md)