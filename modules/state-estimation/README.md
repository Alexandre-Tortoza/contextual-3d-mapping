# State Estimation

`state-estimation` provides motion and pose estimates required to place LiDAR observations in a consistent spatial reference before semantic processing.

The module is independently executable, testable, benchmarkable, and replaceable. Downstream modules depend on its public contracts rather than on a specific odometry implementation.

## Responsibilities

- consume timestamped LiDAR and IMU observations;
- estimate sensor/platform pose and trajectory;
- expose motion-corrected LiDAR observations when supported by the selected backend;
- preserve coordinate-frame, timestamp, uncertainty, and provenance metadata;
- expose health and validity information for downstream consumers.

## Non-responsibilities

- camera-LiDAR calibration or visual correspondence generation;
- learned point embeddings;
- semantic fusion;
- persistent geometric or semantic map construction;
- scene graphs, semantic memory, or contextual reasoning.

Camera-LiDAR association remains owned by `sensor-association`. Learned LiDAR features remain owned by `point-representation`.

## External implementations

Concrete LiDAR-inertial odometry systems are integrated behind adapters. The initial integration target is FAST-LIO, while the public module contracts remain implementation-agnostic so other estimators, simulator ground truth, or dataset-provided poses can be substituted later.

## Initial structure

```text
state-estimation/
├── README.md
├── configs/
├── docs/
├── src/
│   └── state_estimation/
│       ├── application/
│       ├── domain/
│       ├── ports/
│       ├── infrastructure/
│       │   └── fast_lio/
│       └── cli/
├── tests/
└── benchmarks/
```

The package manifest and concrete implementation files should be introduced only when the corresponding implementation issues are addressed.
