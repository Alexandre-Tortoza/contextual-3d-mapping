# Modules

Each module represents an independently developable capability and communicates through shared contracts rather than depending on another module's implementation.

- `visual-perception`: image observations to structured visual features.
- `point-representation`: LiDAR points to learned 3D representations.
- `sensor-association`: geometric and temporal association between sensor observations.
- `semantic-fusion`: multi-source and multi-view fusion into semantic 3D observations.
- `semantic-map`: persistent open-vocabulary spatial representation.
- `semantic-memory`: semantic and spatial retrieval over mapped information.
- `scene-graph`: entities, hierarchy, and relations extracted from the map.
- `context-reasoning`: derives contextual knowledge with explicit provenance.
- `query-engine`: unified semantic, spatial, and contextual query interface.

Each module may later contain its own package configuration, `src/`, `tests/`, `benchmarks/`, and `configs/` as development begins.
