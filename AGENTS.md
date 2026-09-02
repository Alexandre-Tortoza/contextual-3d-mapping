# Agent Guidance

This file defines repository-wide implementation rules for coding agents and contributors.

Read this file before creating or changing code, folders, interfaces, issues, or documentation.

## Architectural intent

This repository intentionally does **not** use Clean Architecture as a repository pattern.

The project uses a simple capability-oriented modular architecture. The goal is to keep the codebase easy to read, navigate, modify, test, and replace while following SOLID principles rigorously where they improve the design.

Do not introduce `domain/`, `application/`, `infrastructure/`, `ports/`, `adapters/`, `repositories/`, or similar architectural layers by default.

Prefer the smallest structure that clearly expresses the responsibility of the module.

## Design priorities

When making architectural decisions, use this order of priority:

1. readability and local understandability;
2. explicit responsibilities;
3. low coupling between capabilities;
4. testability and replaceability;
5. extensibility where a real variation point exists;
6. abstraction only when justified by the previous items.

Do not optimize the architecture for hypothetical future requirements at the expense of current clarity.

## Core repository rules

1. Each capability belongs to one module.
2. Each module owns its implementation details.
3. A module exposes a small public API for consumers.
4. Another module must not depend on private implementation details.
5. Cross-module composition belongs in `apps/` or explicit orchestration code.
6. Shared code must be genuinely repository-wide, not merely reused twice.
7. External libraries and frameworks must not leak their internal data structures across module boundaries when a stable project type is appropriate.
8. New abstractions require a concrete reason.

## Module structure

A module should begin simple. A typical module may evolve toward:

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

Not every directory is mandatory. Create directories only when they have content and a clear purpose.

Do not reproduce the same architectural folder hierarchy inside every module.

## SOLID rules

### Single Responsibility Principle

A module, class, function, or service should have one clear reason to change.

Prefer focused components such as `TemporalFusion`, `ObservationMatcher`, or `PointEncoder` over broad manager or service objects that accumulate unrelated responsibilities.

### Open/Closed Principle

Create a variation point when the project actually needs interchangeable behavior.

Strategies, protocols, factories, or registries are appropriate when multiple implementations exist or are expected by a defined research experiment or integration boundary.

Do not create extension mechanisms speculatively.

### Liskov Substitution Principle

Implementations that satisfy the same public contract must be safely interchangeable by their consumers.

When a contract has multiple implementations, add contract-level tests where practical.

### Interface Segregation Principle

Prefer small interfaces shaped around one consumer need.

Do not create broad interfaces that mix reading, writing, persistence, visualization, reasoning, and export responsibilities.

### Dependency Inversion Principle

High-level orchestration should depend on stable capability contracts rather than concrete external implementations.

For example, orchestration may depend on a `PoseEstimator`, not directly on a specific LiDAR-inertial odometry implementation.

Apply dependency inversion at meaningful boundaries. Do not wrap every class in an interface.

## When to create an interface or protocol

Create one when at least one of these conditions is true:

- multiple implementations already exist;
- the implementation is intentionally replaceable;
- a third-party dependency needs isolation;
- a module boundary requires a stable contract;
- testing requires a lightweight substitute for an expensive or external component.

Avoid interfaces that have only one implementation and no identified boundary or variation need.

## Shared types and contracts

Repository-wide primitives should be minimal and stable. Examples include spatial frames, poses, timestamps, identifiers, and artifact references.

Capability-specific types and contracts belong to the module that defines the capability.

For example, a learned point-feature representation belongs to `point-representation`; it should not be promoted to a global shared package merely because another module consumes it.

A consumer may depend on the producer module's public contract, but not on its private implementation.

## External integrations

Keep an external integration close to the capability that owns it.

Examples:

- a LiDAR-inertial odometry implementation belongs under `state-estimation`;
- a point-cloud processing backend used only by `geometric-map` belongs under that module;
- a model runtime used only by `visual-perception` belongs under that module.

Create repository-wide integration infrastructure only when the integration is genuinely shared by unrelated capabilities.

## Applications

`apps/` contains runnable compositions of capabilities.

Applications may coordinate modules, configure implementations, expose a CLI or GUI, load persisted maps, and connect runtime dependencies.

Applications must not become the owner of research algorithms that belong to modules.

## Research implementations

The architecture is implementation-agnostic.

A paper, model, framework, or external repository may inform an implementation, but the public capability must be named and designed around the project responsibility rather than around the source paper or implementation.

Do not define repository-wide architecture around one algorithm, dataset, model, or research paper.

Issues and architectural contracts should describe the capability and expected behavior. Research references and implementation parallels belong in the relevant module documentation when useful.

## Data flow over hidden coupling

Prefer explicit inputs and outputs over shared mutable global state.

Important transformations should be visible in the code path and represented by named project types where that improves understanding.

Avoid modules reaching into each other's caches, storage layouts, internal model objects, or private configuration.

## Configuration

Configuration should be explicit and typed where practical.

Keep configuration close to the owning capability. Promote settings to application-level configuration only when they participate in composition across modules.

Avoid hidden environment-dependent behavior in core algorithms.

## Error handling

Fail close to the violated boundary with an actionable error message.

Do not silently coerce incompatible coordinate frames, timestamps, tensor shapes, units, map identifiers, or model outputs.

Boundary validation is preferable to debugging corrupted downstream state.

## Testing expectations

Prefer tests at the narrowest useful level:

- unit tests for deterministic local behavior;
- contract tests for interchangeable implementations;
- integration tests for module boundaries;
- end-to-end tests only for representative application flows;
- benchmarks for performance-sensitive research components.

Tests should verify observable behavior rather than internal implementation details unless the internal behavior is itself the research subject.

## Naming

Name components by what they do in this project, not by architectural pattern names.

Prefer:

```text
PointEncoder
PoseEstimator
ObservationMatcher
SemanticMapReader
SemanticMapWriter
```

Avoid vague names such as:

```text
Manager
Helper
Utils
Processor
Service
```

unless the name is genuinely the clearest domain term.

## Documentation

Repository-level decisions belong in `docs/`.

Module implementation details belong in `modules/<module>/docs/`.

When a change introduces a new repository-wide architectural rule, update the relevant root documentation in the same change.

When a change introduces a non-obvious module-local design decision, document the rationale beside that module.

## Decision test before adding complexity

Before adding a layer, abstraction, global package, registry, factory, base class, or interface, answer:

1. What concrete problem does this solve now?
2. Which dependency or responsibility becomes clearer?
3. What future change becomes safer because of it?
4. Could the same result be achieved with a simpler local design?

If these questions do not have clear answers, prefer the simpler design.

## Source of truth

For repository architecture and engineering principles, also read:

- `docs/architecture.md`
- `docs/engineering-principles.md`
- `docs/documentation-policy.md`
- `modules/README.md`

If code and documentation disagree, do not silently choose one. Identify the inconsistency and update the architecture documentation as part of the change when the intended direction is clear.