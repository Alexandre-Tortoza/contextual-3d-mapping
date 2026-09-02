# Documentation

This directory contains the repository-level documentation for `contextual-3d-mapping`.

The documentation here describes the project as a system: its boundaries, repository organization, module topology, integration rules, and end-to-end flow. It intentionally does not document the internal implementation of individual modules.

Module-specific documentation will live beside each module under:

```text
modules/<module>/docs/
```

That local documentation may describe algorithms, model choices, training or inference procedures, configuration, internal architecture, benchmarks, implementation decisions, and module-specific references.

## Repository documentation

- [Architecture](./architecture.md): repository-level architecture, boundaries, and dependency rules.
- [System flow](./system-flow.md): module-to-module flow and integration topology.
- [Documentation policy](./documentation-policy.md): separation between global and module-local documentation.

## Documentation boundary

The root `docs/` directory answers questions such as:

- How is the repository organized?
- How are modules composed?
- What are the allowed dependency directions?
- Where are integration contracts defined?
- How does information move through the system at a high level?

It should not answer questions such as how a specific module implements its capability. Those details belong to that module's own `docs/` directory.
