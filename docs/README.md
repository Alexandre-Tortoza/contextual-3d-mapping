# Pipeline end-to-end do contextual-3d-mapping

Esta documentação acompanha a informação desde os pixels de um frame RGB até a evidência semântica ancorada em geometria 3D. A pergunta central em cada etapa é:

> Que informação existe agora, de onde ela veio, como foi transformada e o que será enviado para a próxima etapa?

A documentação foi reorganizada por estágio da pipeline. O `README.md` é a visão geral; cada arquivo numerado detalha uma transformação específica.

## Regra de documentação

A documentação separa explicitamente três coisas:

```text
arquitetura pretendida
        !=
código implementado
        !=
comportamento realmente observado
```

Sempre que existe evidência versionada, os exemplos usam artifacts reais. Exemplos artificiais não devem ser apresentados como outputs medidos.

## Reference run

A referência visual atual é uma execução real e auditável:

```text
run_id: 20260910T115810Z
git_revision: 7001803
quality_profile: research_quality
frames: 3
reference_frame: corridor-02-000
resolution: 640 x 480
reference_region: region-2c84165423b25fc3
GPU budget: 8.0 GB
observed peak: 4.57 GB
```

Modelos usados nesse run:

| Capacidade | Modelo |
| --- | --- |
| Region discovery | SAM ViT-H |
| Dense visual features | DINOv2-base |
| Language-aligned evidence | CLIP ViT-L/14 |
| Scene/region semantics | Qwen2.5-VL-3B-Instruct, 4-bit |

Esta run é um exemplo do comportamento de uma revisão específica. Ela não é ground truth e não representa garantia de desempenho.

Artifacts principais:

- [`summary.md`](../modules/visual-perception/benchmarks/results/samples/20260910T115810Z/summary.md)
- [`manifest.json`](../modules/visual-perception/benchmarks/results/samples/20260910T115810Z/manifest.json)
- [`diagnostics.json`](../modules/visual-perception/benchmarks/results/samples/20260910T115810Z/frames/corridor-02-000/diagnostics.json)
- [`observation.json`](../modules/visual-perception/benchmarks/results/samples/20260910T115810Z/frames/corridor-02-000/observation.json)
- [`embeddings.npz`](../modules/visual-perception/benchmarks/results/samples/20260910T115810Z/frames/corridor-02-000/embeddings.npz)

## Pipeline geral

```mermaid
flowchart TD
    RGB["01 RGB input<br/>ImageObservation + ImagePayload"]
    RGB --> RD["02 Region Discovery<br/>SAM / RegionProposal"]
    RD --> RM["03 Region Merge<br/>ObservedRegion"]
    RGB --> DF["04 Dense Features<br/>DINOv2 / FeatureMap"]
    RM --> POOL["05 Mask-aware Pooling"]
    DF --> POOL
    POOL --> RE["06 Region Evidence"]
    RM --> LA["07 Language-Aligned Evidence<br/>CLIP"]
    RGB --> SC["08 Scene Context<br/>Qwen2.5-VL"]
    RE --> SEM["09 Region Semantics"]
    LA --> SEM
    SC --> SEM
    SEM --> HS["10 Hypothesis Support"]
    HS --> REF["11 Selective Refinement"]
    REF --> REC["12 Reconciliation"]
    REC --> REL["13 Relations"]
    SC --> VO["14 VisualObservation"]
    REL --> VO

    LIDAR["15 LiDAR / Geometric Map"] --> SA["17 Sensor Association"]
    PC["16 Pose + Calibration"] --> SA
    VO --> SA
    SA --> PVA["18 PointVisualAssociation"]
    PVA --> SF["19 Semantic Fusion"]
    SF --> SM["20 Semantic Map / Memory"]
```

## Estado atual

| Capacidade | Estado | Evidência atual |
| --- | --- | --- |
| `visual-perception` | implementado | run `20260910T115810Z`, artifacts por frame |
| `state-estimation` | primeiro slice implementado | contracts e integração FAST-LIO |
| `geometric-map` | primeiro slice implementado | geometria persistente e referências estáveis |
| `sensor-association` | implementado | projeção, suporte válido e oclusão |
| `semantic-fusion` | implementado no nível de ponto | fusão multi-keyframe e suporte espacial |
| `semantic-map` | planejado | sem schema público concreto |
| `semantic-memory` | planejado | capacidade reservada |
| `scene-graph` | planejado | capacidade reservada |
| `context-reasoning` | planejado | capacidade reservada |
| `query-engine` | planejado | capacidade reservada |

A reference run visual possui artifacts reais até `VisualObservation`. Para a parte 3D existem implementação e diagnósticos reais, mas ainda não há um único artifact versionado que acompanhe `region-2c84165423b25fc3` até um `GeometryReference` e depois até uma entidade persistente. A documentação não inventa essa continuidade.

## Documentação por estágio

1. [Entrada RGB](./01-input-rgb.md)
2. [Region Discovery](./02-region-discovery.md)
3. [Region Merge / Consolidation](./03-region-merge.md)
4. [Dense Feature Extraction](./04-dense-features.md)
5. [Mask-aware Pooling](./05-mask-aware-pooling.md)
6. [Region Evidence](./06-region-evidence.md)
7. [Language-Aligned Evidence](./07-language-aligned-evidence.md)
8. [Scene Context](./08-scene-context.md)
9. [Region Semantics](./09-region-semantics.md)
10. [Hypothesis Support](./10-hypothesis-support.md)
11. [Selective Refinement](./11-selective-refinement.md)
12. [Reconciliation intra-frame](./12-reconciliation.md)
13. [Relations](./13-relations.md)
14. [VisualObservation](./14-visual-observation.md)
15. [Geometric Map](./15-geometric-map.md)
16. [Pose + Calibration](./16-pose-calibration.md)
17. [Sensor Association](./17-sensor-association.md)
18. [PointVisualAssociation](./18-point-visual-association.md)
19. [Semantic Fusion](./19-semantic-fusion.md)
20. [Semantic Map / Memory](./20-semantic-map.md)

## Como cada página deve ser mantida

Cada estágio deve responder, quando aplicável: objetivo, entrada, contract, transformação, tecnologias, exemplo da reference run, artifacts, saída, consumidor, limitações atuais e referências científicas. Quando não houver artifact real, a página deve declarar isso explicitamente.

A documentação anterior foi preservada em [`.old-docs/`](../.old-docs/) para consulta histórica.