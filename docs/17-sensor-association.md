# 17. Sensor Association

## Onde estamos na pipeline

```mermaid
flowchart LR
    A["VisualObservation / RGB"] --> D["Sensor Association"]:::current
    B["GeometryPoint"] --> D
    C["Pose + Calibration"] --> D
    D --> E["PointVisualAssociation"]
    classDef current stroke-width:3px,font-weight:bold;
```

## Objetivo

Responder, para cada ponto 3D candidato:

> Se este ponto fosse observado por esta câmera neste instante, onde cairia na imagem e qual evidência visual válida cobre esse pixel?

É aqui que uma região 2D finalmente pode tocar a geometria 3D.

## Entradas

A etapa combina quatro fontes:

```text
GeometryPoint
    -> XYZ persistente

pose
    -> relação entre mapa e câmera naquele instante

CameraLidarCalibration
    -> modelo óptico e extrínseca

VisualObservation
    -> masks, region_id, claims e evidências visuais
```

## Fluxo

```text
ponto XYZ
 -> transform para camera
 -> projeção calibrada
 -> pixel (u,v)
 -> teste de hemisfério/frente
 -> bounds da imagem
 -> suporte óptico válido
 -> oclusão
 -> membership de máscara
 -> cor + region_id + label + feature_reference
```

## Exemplo conceitual

Suponha que a etapa anterior projete um ponto para:

```text
pixel = (420, 290)
```

Agora verificamos:

```text
1. está na frente da câmera?
2. está dentro de 640 x 480?
3. está dentro da área válida do fisheye?
4. existe outro ponto mais próximo ocupando esse suporte visual?
5. qual máscara contém (420, 290)?
```

Se o pixel cair dentro da máscara da região `region-abc`:

```text
GeometryReference
        |
        + pixel (420,290)
        + region_id = region-abc
        + RGB observado
        + evidência semântica da região
        |
        v
PointVisualAssociation
```

## Membership de máscara

A associação não precisa perguntar ao Qwen novamente qual objeto está naquele pixel.

A semântica já foi produzida em `visual-perception`. O que esta etapa faz é descobrir **qual região visual cobre o pixel projetado**.

```text
pixel projetado
    -> consultar masks de ObservedRegion
    -> identificar region_id
    -> recuperar evidência daquela região
```

Isso mantém percepção e geometria separadas.

## Estados de rejeição

Nem todo ponto projetado vira associação.

Estados explícitos incluem:

```text
behind_camera
outside_image
outside_valid_support
occluded
associated
```

Uma rejeição não deve carregar evidência visual residual.

## Oclusão

Oclusão responde:

```text
este ponto realmente é visível
ou está atrás de outra superfície?
```

Sem esse teste, um ponto atrás de uma parede pode cair no mesmo pixel de uma região visível e receber sua semântica incorretamente.

## Scan isolado vs mapa persistente

O módulo expõe duas fronteiras:

```text
associate_points
    scan LiDAR recém-chegado

associate_map_points
    geometria persistida
```

Um scan visto de um único viewpoint permite z-buffer por pixel mais direto.

Um mapa acumulado é esparso. Entre amostras da parede frontal podem surgir lacunas de pixel que deixam pontos do fundo "vazar".

Por isso `associate_map_points` usa uma política mais conservadora com células de profundidade e vizinhança.

## Diagnóstico real do corridor-02

Uma versão anterior da associação do mapa persistente usava z-buffer somente no pixel exato.

Em mapa esparso isso permitia:

```text
parede frontal
. . . . .   <- amostras esparsas
    x        <- lacuna

ponto atrás da parede
    -> projeta na lacuna
    -> parece visível
    -> recebe label da imagem frontal
```

O efeito observado era classificar pontos através de paredes, incluindo pontos em altura de parede recebendo `ceiling`.

A correção passou a usar células de pixels com consulta de vizinhança `3x3` e profundidade no eixo óptico.

A política é conservadora:

```text
preferir perder alguns pontos de borda
em vez de contaminar o fundo
```

## Profundidade no eixo óptico

Em câmeras wide-angle, usar distância radial pode causar falsos testes de oclusão na periferia.

Por isso a comparação usa profundidade no eixo óptico da câmera.

```text
radial distance != optical-axis depth
```

Essa distinção é especialmente importante no fisheye do corridor-02.

## Outro problema real: geometria duplicada

Uma abordagem anterior anexava scans coloridos ao mapa persistente.

Isso criava duas amostragens próximas da mesma superfície:

```text
ponto persistente cinza
+
ponto do scan colorido
```

No viewer apareciam pontos cinzas "fantasmas" ao lado dos coloridos.

A associação ancorada diretamente na geometria persistente evita essa duplicação conceitual.

## Reference run

A run visual `20260910T115810Z` não persiste uma linha de `PointVisualAssociation` para `region-2c84165423b25fc3`.

Os exemplos numéricos desta página são, portanto, conceituais. O diagnóstico de oclusão do corridor-02, porém, vem de comportamento real medido do pipeline 3D.

## Próxima evolução

Hoje a associação pode transportar referências de evidência regional.

Para point representation e distilação cross-modal, o caminho mais rico será:

```text
GeometryPoint
 -> pixel (u,v)
 -> posição na FeatureMap DINO
 -> feature visual local [768]
 -> associação ao ponto 3D
```

Isso evita atribuir exatamente o mesmo embedding agregado a todos os pontos de uma região.

## Próxima leitura

- [18. PointVisualAssociation](./18-point-visual-association.md)
- [19. Semantic Fusion](./19-semantic-fusion.md)
- [Documentação de `sensor-association`](../modules/sensor-association/docs/README.md)