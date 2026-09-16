# Region Discovery Flow

Este documento detalha os dois caminhos de descoberta geométrica que convergem antes da filtragem e do merge: discovery genérico e concept-conditioned grounding.

## Discovery genérico + grounding por conceito

```mermaid
flowchart TD
    RGB[RGB Frame]

    RGB --> GENERIC[Generic Region Discovery]
    GENERIC --> PASSES[full-frame + optional overlapping tiles]
    PASSES --> SAMT[SAM3 Tracker]
    SAMT --> GMASK[generic masks]
    GMASK --> REMAP[remap tile coordinates when needed]
    REMAP --> GP[generic RegionProposal array]

    RGB --> SCD[SceneConceptDiscoverer: Qwen / Gemini]
    SCD --> RAW[entities + contextual_features]
    RAW --> NORM[normalize + deduplicate]
    NORM --> FILTER[discard standalone structural surfaces]
    FILTER --> PRIOR[prioritize contextual concepts + max_concepts]
    PRIOR --> CONCEPTS[SceneConceptSet]

    RGB --> ENCODE[SAM3 PCS image encoding]
    CONCEPTS --> QUERY[query each concept]
    ENCODE --> QUERY
    QUERY --> CMASK[concept-conditioned masks]
    CMASK --> CP[concept RegionProposal array]

    GP --> UNION[union proposals]
    CP --> UNION

    UNION --> AREA[proposal filtering]
    AREA --> VALID[valid lens area]
    VALID --> EGO[ego / rig exclusion]
    EGO --> QUALITY[geometry + area + quality rules]
    QUALITY --> ACCEPT[accepted proposals]
    QUALITY --> REJECT[RejectedProposal array]

    ACCEPT --> MERGE[cross-scale merge]
    MERGE --> DEDUP[resolve overlaps / duplicates]
    DEDUP --> REGION[ObservedRegion array]
    REGION --> FREEZE[GEOMETRY FREEZE]
```

## Tiling

```mermaid
flowchart TD
    RGB[RGB Frame] --> FULL[full-frame pass]
    RGB --> TILES[overlapping tiles]
    FULL --> SAM1[SAM3 Tracker]
    TILES --> LOOP[for each tile]
    LOOP --> SAM2[SAM3 Tracker]
    SAM2 --> BORDER{discard tile border truncations?}
    BORDER -->|sim| TOUCH{proposal touches internal tile border?}
    TOUCH -->|sim| DROP[discard truncated proposal]
    TOUCH -->|não| REMAP[remap to global frame]
    BORDER -->|não| REMAP
    SAM1 --> FP[full-frame proposals]
    REMAP --> TP[tile proposals]
    FP --> UNION[generic proposal set]
    TP --> UNION
```

## SAM3 em dois papéis

```mermaid
flowchart LR
    SAM3[SAM3]
    SAM3 --> TRACKER[Tracker / mask generation]
    TRACKER --> CLASSLESS[descoberta genérica class-agnostic]
    SAM3 --> PCS[Promptable Concept Segmentation]
    PCS --> GROUNDED[conceito conhecido -> localização / máscara]
    CLASSLESS --> SET[proposal set]
    GROUNDED --> SET
```

O grounding semântico adiciona proposals ao discovery genérico. Ele não substitui o caminho class-agnostic. Ambos passam pelas mesmas políticas de área e pelo mesmo merge geométrico.