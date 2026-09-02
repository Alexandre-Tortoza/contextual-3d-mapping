# Geometric Map

`geometric-map` owns persistent world geometry assembled from contract-compatible motion estimates and LiDAR observations.

## Responsibilities

- consume pose, trajectory, and LiDAR observations through public contracts;
- place observations in a consistent world frame;
- maintain persistent geometric map state;
- expose stable geometry identifiers and spatial bounds;
- preserve geometry provenance and source-observation references;
- provide geometry suitable for downstream semantic association, mapping, evaluation, and visualization.

## Non-responsibilities

- estimating motion from raw LiDAR/IMU inputs;
- learned point representation;
- visual perception;
- semantic classification or fusion;
- semantic memory, scene graphs, or contextual reasoning;
- user-facing rendering.

`state-estimation` supplies motion context. `semantic-map` enriches geometry through stable references rather than duplicating geometric ownership.

## Initial structure

```text
geometric-map/
├── README.md
├── configs/
├── docs/
├── src/
│   └── geometric_map/
│       ├── application/
│       ├── domain/
│       ├── ports/
│       └── infrastructure/
├── tests/
└── benchmarks/
```

Concrete data structures, indexes, reconstruction strategies, persistence formats, and implementations should be introduced by implementation issues while keeping public contracts implementation-agnostic.
