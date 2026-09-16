# 04. Dense Feature Extraction

## Onde estamos na pipeline

```mermaid
flowchart LR
    A["RGB"] --> B["DINOv2"]:::current
    B --> C["FeatureMap Hf x Wf x C"]
    C --> D["Mask-aware Pooling"]
    classDef current stroke-width:3px,font-weight:bold;
```

## Objetivo

Transformar a imagem em uma grade espacial de vetores visuais densos. DINOv2 não retorna labels. Cada posição da feature map contém um embedding que representa aparência e estrutura visual aprendidas pelo backbone.

Uma forma útil de pensar no DINO é:

```text
imagem
 -> pequenos blocos visuais (patches)
 -> um vetor para cada patch
 -> grade espacial de vetores
```

## Patches na configuração atual

O checkpoint da reference run é `facebook/dinov2-base`. Esse modelo usa patch size `14 x 14` no espaço processado.

O frame real possui:

```text
640 x 480 pixels
```

A configuração usa `input_resolution: 448`. O adapter preserva aspect ratio, limita a maior aresta a aproximadamente 448 e força dimensões múltiplas de 14.

Para esse frame:

```text
original:    640 x 480
processado:  448 x 336
patch size:   14 x 14
```

Logo:

```text
448 / 14 = 32 patches na largura
336 / 14 = 24 patches na altura
```

A grade espacial produzida pelo DINO é, portanto, conceitualmente:

```text
24 x 32 patches
```

DINOv2-base produz um embedding de `768` dimensões por patch.

Então o tensor final tem shape:

```text
FeatureMap.shape = [24, 32, 768]
```

Esse é um tensor 3D no sentido:

```text
[altura espacial, largura espacial, dimensão do vetor]
```

Não significa XYZ do mundo físico.

## Como é um vetor de feature

Para uma posição específica da grade:

```text
FeatureMap[8, 17]
```

podemos imaginar algo como:

```text
[0.031, -0.182, 0.044, 0.217, ..., 0.092]
```

com `768` números.

Esses valores não correspondem diretamente a campos humanos como:

```text
[door=0.8, wall=0.1, floor=0.1]
```

O vetor representa uma posição em um espaço aprendido. Patches com aparência e estrutura visual semelhantes tendem a ficar mais próximos nesse espaço.

## O token CLS não entra na grade

O modelo também pode produzir tokens prefixados, como CLS e register tokens. O adapter separa esses tokens dos tokens espaciais antes de reconstruir a grade.

```text
output do transformer
├── prefix tokens
└── spatial tokens
       |
       v
reshape -> [24, 32, 768]
```

Isso é importante porque o pipeline precisa preservar a correspondência espacial entre posição da imagem e posição da feature map.

## Relação com a máscara do SAM

DINO analisa a imagem inteira. SAM, em paralelo, define regiões.

Depois os dois sinais se encontram:

```text
SAM
 -> máscara da região

DINO
 -> FeatureMap [24, 32, 768]

máscara + FeatureMap
 -> Mask-aware Pooling
 -> embedding visual da região [768]
```

Ou seja, o SAM responde **onde está a região** e o DINO responde **como as partes visuais dessa região são representadas**.

## DINO não fornece a label ao Qwen

Na pipeline atual, o vetor DINO não é enviado diretamente ao Qwen para que o Qwen gere a label.

O Qwen interpreta views em pixels derivadas da máscara do SAM e recebe contexto estruturado da cena. O DINO permanece como uma fonte separada de evidência visual densa.

Isso é importante para entender a arquitetura:

```text
SAM mask -> views em pixels -> Qwen -> hipóteses semânticas

RGB -> DINO -> dense features -> pooling -> evidência visual densa
```

Mais tarde essas evidências podem ser usadas para coerência, reconciliação e futura projeção 2D para 3D.

## Reference run

NOTE: exemplo conceitual — os artifacts da run que originalmente acompanhava esta configuração foram removidos do repositório (ver nota em [`docs/README.md`](./README.md#reference-run)).

```text
backend: dinov2
checkpoint: facebook/dinov2-base
input_resolution: 448
upsampling: nearest
dimension: 768
modality: visual_dense
normalized: true
```

## O que high-resolution significa hoje

O caminho observado nessa run usa `pixel_nearest_highres`: a máscara é consultada contra a feature map densa usando alinhamento espacial e nearest sampling.

Isso **não cria detalhes novos entre patches**. O DINO continua tendo sua resolução nativa de tokens. Upsampling apenas oferece uma forma mais conveniente de consultar a representação em coordenadas de imagem.

Exemplo conceitual:

```text
patch DINO representa uma área de imagem
        |
nearest/bilinear
        |
consultas em posições mais densas
```

O conteúdo semântico do vetor continua vindo do backbone, não da interpolação.

## Saída

`FeatureMap` é combinada com a máscara de cada `ObservedRegion` em [05. Mask-aware Pooling](./05-mask-aware-pooling.md).

## Impacto no mapa contextual

Quanto melhor a granularidade e estabilidade das features, mais precisa pode ser a associação entre evidência visual e uma região.

No futuro, em vez de usar apenas um embedding agregado da região, o sistema poderá projetar uma feature visual próxima ao pixel de cada ponto LiDAR:

```text
ponto XYZ
 -> pixel (u,v)
 -> posição na FeatureMap DINO
 -> vetor 768D associado ao ponto
```

Isso permitiria comparar coerência visual e estrutural diretamente no mapa 3D.

Para o fluxo completo, consulte [Pipeline detalhada de Visual Perception](../modules/visual-perception/docs/pipeline.md).

## Próxima leitura

- [05. Mask-aware Pooling](./05-mask-aware-pooling.md)
- [Pipeline detalhada de `visual-perception`](../modules/visual-perception/docs/pipeline.md)