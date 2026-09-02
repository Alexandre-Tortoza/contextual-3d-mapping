# System Flow

This document shows the repository-level flow between modules. It intentionally represents only composition and data movement between boundaries. It does not describe how any module performs its work internally.

Arrows between modules represent exchange through compatible contracts, not direct implementation dependencies.

```mermaid
flowchart LR
    D[Datasets] --> A[adapters]
    S[Live or recorded sensors] --> A

    A --> SE[state-estimation]
    A --> VP[visual-perception]

    SE -->|motion-corrected LiDAR| PR[point-representation]
    SE -->|pose / trajectory| SA[sensor-association]

    VP --> SA
    PR --> SA

    SA --> SF[semantic-fusion]
    SF --> SMAP[semantic-map]

    SMAP --> SMEM[semantic-memory]
    SMAP --> SG[scene-graph]

    SMEM --> CR[context-reasoning]
    SG --> CR

    SMEM --> QE[query-engine]
    SG --> QE
    CR --> QE

    QE --> APP[apps]

    C[contracts] -. shared interfaces .-> A
    C -. shared interfaces .-> SE
    C -. shared interfaces .-> VP
    C -. shared interfaces .-> PR
    C -. shared interfaces .-> SA
    C -. shared interfaces .-> SF
    C -. shared interfaces .-> SMAP
    C -. shared interfaces .-> SMEM
    C -. shared interfaces .-> SG
    C -. shared interfaces .-> CR
    C -. shared interfaces .-> QE
```

## Interpretation

The diagram is a topology of the intended integration path, not a specification of internal algorithms and not a requirement that every runnable experiment use every module.

`state-estimation` establishes the motion and pose context required by LiDAR-based processing. It may provide motion-corrected LiDAR frames to `point-representation` and pose or trajectory information to `sensor-association`. Camera-LiDAR calibration and visual-to-point correspondence remain outside state estimation.

A workflow may replace, isolate, or omit stages when its contracts permit that composition. For example, an experiment may inject simulator or dataset ground-truth poses instead of running a live state estimator. The important invariant is that module boundaries remain explicit and that interoperability is expressed through repository contracts.

`evaluation/`, `experiments/`, and `tests/` are intentionally not represented as sequential stages. They are cross-cutting consumers that may exercise individual modules or complete compositions independently.
