# Modules

Each module represents an independently developable capability and communicates through shared contracts rather than depending on another module's implementation.

- `state-estimation`: LiDAR/IMU observations to pose, trajectory, and motion-corrected LiDAR frames.
- `geometric-map`: persistent world geometry built from contract-compatible poses and LiDAR observations.
- `visual-perception`: image observations to structured visual features.
- `point-representation`: LiDAR points to learned 3D representations.
- `sensor-association`: geometric and temporal association between sensor observations.
- `semantic-fusion`: multi-source and multi-view fusion into semantic 3D observations.
- `semantic-map`: persistent open-vocabulary semantic information linked to world geometry.
- `semantic-memory`: semantic and spatial retrieval over mapped information.
- `scene-graph`: entities, hierarchy, and relations extracted from the map.
- `context-reasoning`: derives contextual knowledge with explicit provenance.
- `query-engine`: unified semantic, spatial, and contextual query interface.

Each module may contain its own package configuration, `src/`, `tests/`, `benchmarks/`, `configs/`, and `docs/` as development begins.

Persistent geometric reconstruction belongs to `geometric-map`, not to `state-estimation` or `semantic-map`. Concrete odometry implementations remain internal to `state-estimation`, while semantic modules reference geometry through public contracts.
