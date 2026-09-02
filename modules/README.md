# Modules

Each module represents one independently understandable, developable, testable, benchmarkable, and replaceable capability.

The repository is capability-oriented. Modules do not need to reproduce Clean Architecture layers such as `domain/`, `application/`, `infrastructure/`, `ports/`, or `adapters/`.

Prefer the smallest internal structure that makes the capability clear.

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

Persistent geometric reconstruction belongs to `geometric-map`, not to `state-estimation` or `semantic-map`.

Concrete odometry implementations remain internal to `state-estimation`. Semantic modules reference geometry through documented public types instead of owning duplicate authoritative geometry.

## Public boundary

Each module should expose a small documented public API.

Other modules may depend on that public API but must not depend on private implementation details, internal model objects, storage layout, caches, training-only structures, or backend-specific representations.

Capability-specific contracts belong to the capability that defines them.

For example, a learned point representation contract belongs to `point-representation`, even if `sensor-association` or another module consumes it.

## Internal structure

A module may evolve toward a structure such as:

```text
modules/<module>/
├── README.md
├── src/
│   └── <package>/
├── tests/
├── configs/
├── benchmarks/
└── docs/
```

None of these directories is mandatory before it has a concrete purpose.

Keep capability-specific external integrations close to the module that owns them. Do not move them into a repository-wide infrastructure layer merely because they wrap a third-party library.

## Abstractions

Protocols, interfaces, strategies, factories, or registries should exist only for concrete boundaries or variation points, for example:

- multiple implementations;
- intentional replaceability;
- third-party dependency isolation;
- module boundary stability;
- contract-level testing with substitutes.

Do not create an interface for every class or add layers speculatively.

## Documentation

Detailed module behavior, algorithms, implementation rationale, model choices, benchmarks, limitations, and research references belong under:

```text
modules/<module>/docs/
```

Repository-wide architecture decisions belong under root `docs/`.

Before implementing a module, read the root [`AGENTS.md`](../AGENTS.md), [`docs/architecture.md`](../docs/architecture.md), and [`docs/engineering-principles.md`](../docs/engineering-principles.md).