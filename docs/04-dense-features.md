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

Transformar a imagem em uma grade espacial de vetores visuais densos. DINOv2 não retorna labels. Cada posição da feature map contém um embedding que representa aparência e estrutura visual condicionadas pelo frame.

## Transformação

```text
RGB
 -> resize/preprocess
 -> patches
 -> DINOv2
 -> tokens espaciais
 -> FeatureMap [Hf, Wf, C]
```

Os tokens especiais são separados dos tokens espaciais antes da reconstrução da grade. A feature map declara stride, dimensão, checkpoint, representação, interpolação e suporte válido.

## Reference run

```text
backend: dinov2
checkpoint: facebook/dinov2-base
input_resolution: 448
upsampling: nearest
dimension: 768
modality: visual_dense
normalized: true
```

O run de referência usa DINOv2-base. A região acompanhada é `region-2c84165423b25fc3`.

## O que high-resolution significa hoje

O caminho observado nesta run usa `pixel_nearest_highres`: a máscara é consultada contra a feature map densa, mas isso não equivale a reconstruir informação visual inexistente entre patches. Outros caminhos do código podem usar sampling/interpolação ou learned upsampling, mas a referência acima deve refletir o run efetivamente medido.

## Saída

`FeatureMap` é combinada com a máscara de cada `ObservedRegion` em [05. Mask-aware Pooling](./05-mask-aware-pooling.md).

## Impacto no mapa contextual

Quanto melhor a granularidade e estabilidade das features, mais precisa pode ser a associação entre evidência visual e uma região. Para futura supervisão 2D -> 3D, a granularidade por posição é ainda mais importante porque o alvo deixa de ser apenas um embedding agregado da região e pode se tornar uma feature alinhada a cada ponto LiDAR.

## Referências científicas

DINOv2 é a base atual das dense visual features. A documentação histórica sobre comparação de alta resolução e literatura relacionada foi preservada em `.old-docs/`.