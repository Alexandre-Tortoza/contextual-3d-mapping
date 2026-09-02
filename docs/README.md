# Documentation

This directory contains repository-level documentation for `contextual-3d-mapping`.

The documentation here describes the project as a system: its architectural intent, module boundaries, engineering principles, integration rules, application composition, persistence boundaries, and end-to-end flow.

The project intentionally uses a simple capability-oriented modular architecture rather than Clean Architecture as a repository-wide pattern.

Detailed implementation documentation lives beside each module under:

```text
modules/<module>/docs/
```

That local documentation may describe algorithms, model choices, training or inference procedures, configuration, internal architecture, benchmarks, implementation decisions, limitations, and module-specific research references.

## Repository documentation

- [Architecture](./architecture.md): repository-level architecture, capability ownership, boundaries, and dependency rules.
- [Engineering principles](./engineering-principles.md): implementation decisions, SOLID interpretation, abstraction thresholds, integration locality, testing, and review guidance.
- [System flow](./system-flow.md): module-to-module and application flow.
- [Applications](./applications.md): responsibilities and boundaries of runnable applications.
- [Map lifecycle](./map-lifecycle.md): lifecycle from observations to persisted and queryable maps.
- [Documentation policy](./documentation-policy.md): separation between global and module-local documentation.
- [`AGENTS.md`](../AGENTS.md): condensed repository rules that coding agents and contributors must read before changing architecture or code.

## Architecture summary

The repository follows a small set of stable rules:

1. each capability has one obvious module owner;
2. modules expose small public APIs and hide implementation details;
3. applications compose capabilities rather than owning research algorithms;
4. capability-specific integrations and contracts stay close to their owning modules;
5. shared code is limited to genuinely repository-wide, stable primitives;
6. abstractions are introduced for concrete boundaries or variation points, not speculatively;
7. SOLID guides code and dependency design, not folder naming;
8. external papers, models, repositories, and datasets may influence implementations without defining the public architecture.

## Documentation boundary

The root `docs/` directory answers questions such as:

- Why is the repository organized this way?
- How are modules composed?
- What capability owns a particular responsibility?
- What are the allowed dependency directions?
- When should a shared contract or abstraction exist?
- How does information move through the system at a high level?
- Which application builds maps and which application explores them?
- Where does persistence belong architecturally?
- Which engineering rules should remain stable as research implementations change?

It should not explain the internal algorithm of a specific module. Those details belong to that module's own `docs/` directory.