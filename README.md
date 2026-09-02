# contextual-3d-mapping

Research framework for building open-vocabulary 3D semantic and contextual maps from RGB, LiDAR, and motion estimates, combining persistent world geometry, visual-language features, learned point representations, semantic memory, scene graphs, and spatial reasoning.

## Architecture

The repository uses a **simple capability-oriented modular architecture**.

The main goals are readability, explicit responsibility, low coupling, testability, replaceability, maintainability, and scientific reproducibility. SOLID principles guide code and dependency design at meaningful boundaries.

The central rule is simple:

```text
one capability -> one clear owner module -> small public API -> explicit composition
```

Capability-specific integrations and implementation details stay with the module that owns them. Applications compose modules into runnable workflows. Shared primitives remain small and stable.

## High-level flow

```text
RGB + LiDAR + IMU
        -> state-estimation
        -> geometric-map
        -> visual-perception / point-representation
        -> sensor-association
        -> semantic-fusion
        -> semantic-map
        -> semantic-memory / scene-graph / context-reasoning
        -> query-engine
        -> applications
```

`apps/mapping-runtime` composes map construction workflows. `apps/map-explorer` opens persisted maps for 3D visualization, query, evidence inspection, and graph exploration. `apps/cli` provides scriptable access to application-level operations.

Coding agents and contributors should read [`AGENTS.md`](./AGENTS.md) before creating or changing code, folders, interfaces, or repository-wide architecture.

## Documentation

Repository-level architecture and integration documentation is available in [`docs/README.md`](./docs/README.md).

Important architecture decisions are documented in:

- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/system-flow.md`](./docs/system-flow.md)
- [`docs/applications.md`](./docs/applications.md)
- [`docs/map-lifecycle.md`](./docs/map-lifecycle.md)
- [`docs/engineering-principles.md`](./docs/engineering-principles.md)
- [`docs/documentation-policy.md`](./docs/documentation-policy.md)

Detailed implementation documentation lives inside each module under `modules/<module>/docs/` as those modules are developed.
