# 15. Geometric Map

## Onde estamos na pipeline

```mermaid
flowchart LR
    A["LiDAR + IMU"] --> B["state-estimation"]
    B --> C["motion-corrected LiDAR + StateEstimate"]
    C --> D["geometric-map"]:::current
    D --> E["GeometryPoint / MapAnchoredPoint"]
    E --> F["Sensor Association"]
    classDef current stroke-width:3px,font-weight:bold;
```

## Objetivo

Fornecer a geometria autoritativa e persistente do mundo. `visual-perception` descreve pixels e regiões; ela não cria coordenadas XYZ.

## Contract conceitual

```text
GeometryPoint
{
    reference: GeometryReference
    coordinates_m: (x, y, z)
    source_coordinates_m
    source_observation
    provenance
}
```

`GeometryReference` fornece identidade estável dentro de um mapa persistente.

## Relação com contexto

A geometria é o suporte onde as evidências de múltiplas observações podem convergir. Sem uma referência persistente, labels de frames diferentes continuam sendo observações independentes.

## Reference run

A reference run visual `20260910T115810Z` não contém um artifact único que ligue `region-2c84165423b25fc3` a um `GeometryReference` concreto. Portanto esta página não cria coordenadas de exemplo como se fossem medidas.

Existem implementação e workflows reais do `geometric-map`, além de integração com state estimation, mas a evidência visual de referência termina em `VisualObservation`.

## Point representation planejada

O módulo `point-representation` foi reservado para adicionar embeddings aprendidos por ponto. Essa representação complementará a geometria, não a substituirá:

```text
GeometryPoint -> onde o ponto está
PointEmbedding -> como a estrutura local 3D é representada
Semantic evidence -> o que as observações afirmam sobre ela
```

Essa separação é importante para futuras estratégias de coerência 3D e distilação cross-modal.