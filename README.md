# contextual-3d-mapping

Research framework for building open-vocabulary 3D semantic and contextual maps from RGB, LiDAR, and motion estimates, combining visual-language features, learned point representations, semantic memory, scene graphs, and spatial reasoning.

## Architecture

The repository uses a **simple capability-oriented modular architecture**. It intentionally does not adopt Clean Architecture as a repository-wide pattern.

The main goals are readability, explicit responsibility, low coupling, testability, replaceability, and maintainability. SOLID principles guide code and dependency design without imposing unnecessary folder layers or abstractions.

Coding agents and contributors should read [`AGENTS.md`](./AGENTS.md) before creating or changing code, folders, interfaces, or repository-wide architecture.

## Documentation

Repository-level architecture and integration documentation is available in [`docs/README.md`](./docs/README.md).

Important architecture decisions are documented in:

- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/engineering-principles.md`](./docs/engineering-principles.md)
- [`docs/documentation-policy.md`](./docs/documentation-policy.md)

Detailed implementation documentation lives inside each module under `modules/<module>/docs/` as those modules are developed.