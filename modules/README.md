# Modules

Each module represents one independently understandable, developable, testable, benchmarkable, and replaceable capability.

The repository is capability-oriented. Prefer the smallest internal structure that makes the capability, its inputs, outputs, and variation points clear.

## Capability ownership

- `state-estimation`: LiDAR/IMU observations to pose, trajectory, and motion-corrected LiDAR frames.
- `geometric-map`: persistent world geometry built from poses and geometric observations.
- `visual-perception`: image observations to structured visual and semantic features.
- `point-representation`: LiDAR points to learned 3D representations.
- `sensor-association`: geometric and temporal association between sensor observations.
- `semantic-fusion`: multi-source, temporal, and multi-view fusion into semantic 3D observations.
- `semantic-map`: persistent open-vocabulary semantic information linked to world geometry.
- `semantic-memory`: semantic and spatial retrieval over mapped information.
- `scene-graph`: entities, hierarchy, and relations extracted from the map.
- `context-reasoning`: contextual inference with explicit provenance.
- `query-engine`: unified semantic, spatial, and contextual query interface.

Persistent geometric reconstruction is owned by `geometric-map`.

Concrete odometry implementations are owned by `state-estimation`. Semantic modules reference geometry through documented public types so the system keeps one authoritative geometric representation.

## Public boundary

Each module exposes a small documented public API.

Consumers depend on:

- public data types;
- public functions or classes;
- stable entry points;
- protocols that represent real variation points.

Implementation-specific model objects, caches, storage layout, training structures, and backend representations remain local to the owning module.

Capability-specific contracts belong to the capability that defines them.

For example, a learned point representation contract belongs to `point-representation`, even when `sensor-association` consumes it.

## Internal structure

A module may evolve toward:

```text
modules/<module>/
├── README.md
├── src/
│   └── <package>/
│       ├── __init__.py
│       ├── models.py
│       ├── config.py
│       └── <capability files>.py
├── tests/
├── configs/
├── benchmarks/
└── docs/
```

Create each directory when it has a concrete responsibility and content.

Keep capability-specific external integrations close to the module that owns them. For example, a LiDAR-inertial odometry backend belongs under `state-estimation`, and a point-cloud backend used only for reconstruction belongs under `geometric-map`.

## Implementation sequence

When developing a module:

1. state the capability responsibility in `README.md`;
2. define public inputs and outputs;
3. document units, frames, timestamps, shapes, provenance, and other boundary invariants;
4. implement the simplest working behavior locally;
5. introduce protocols only for actual replacement or comparison points;
6. test local behavior and public contracts;
7. add benchmarks for performance-sensitive behavior;
8. document non-obvious algorithmic and research decisions under `docs/`.

## Abstractions

Use protocols, interfaces, strategies, factories, or registries when they represent a concrete variation point, such as:

- multiple implementations;
- intentional replaceability;
- research comparisons;
- third-party dependency isolation;
- module boundary stability;
- contract-level testing with substitutes.

For straightforward local behavior with one implementation, direct construction and direct calls are preferred because they make the code path easier to read.

## Documentation

Detailed module behavior, algorithms, implementation rationale, model choices, benchmarks, limitations, and research references belong under:

```text
modules/<module>/docs/
```

Repository-wide architecture decisions belong under root `docs/`.

Before implementing a module, read the root [`AGENTS.md`](../AGENTS.md), [`docs/architecture.md`](../docs/architecture.md), and [`docs/engineering-principles.md`](../docs/engineering-principles.md).