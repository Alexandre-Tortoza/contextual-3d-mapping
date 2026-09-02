# Agent Guidance

This file defines repository-wide implementation rules for coding agents and contributors.

Read this file before creating or changing code, folders, interfaces, issues, or documentation.

## Architectural direction

Use a **capability-oriented modular architecture**.

Organize the system around clear capabilities such as state estimation, geometric mapping, visual perception, point representation, sensor association, semantic fusion, semantic memory, scene graphs, contextual reasoning, and querying.

The implementation goal is simple: a reader should be able to open one module and understand most of that capability without exploring unrelated directories.

Apply SOLID principles at meaningful code and module boundaries while keeping the structure direct and easy to navigate.

## Design priorities

When making architectural decisions, use this order of priority:

1. readability and local understandability;
2. explicit responsibility and ownership;
3. low coupling between capabilities;
4. testability and replaceability;
5. extensibility around real variation points;
6. abstraction when it makes one of the previous properties clearer or safer.

Prefer designs whose intent is visible from file names, types, function signatures, and data flow.

## Implementation workflow

For every new capability or change:

1. identify the module that owns the behavior;
2. define the inputs, outputs, invariants, units, frames, and provenance that cross the boundary;
3. implement the behavior inside the owning module;
4. expose only the public types and operations consumers need;
5. compose modules from `apps/`, experiments, or explicit orchestration code;
6. add tests at the narrowest useful level;
7. document non-obvious decisions beside the code that owns them.

If ownership is unclear, resolve that first. Ownership is an architectural decision.

## Repository roles

Use the repository directories as follows:

```text
apps/         runnable compositions and user-facing entry points
modules/      research and system capabilities
datasets/     dataset manifests, schemas, splits, and dataset-level support
evaluation/   reusable metrics and evaluation logic
experiments/  comparisons, ablations, and experiment orchestration
docs/         repository-wide architecture and decisions
tests/        repository-level integration or end-to-end tests
```

When repository-wide primitives become necessary, keep them small and stable in a shared location. Good examples are timestamps, spatial frames, poses, transforms, map identifiers, and provenance primitives.

Capability-specific integrations, contracts, configuration, persistence helpers, and model runtimes stay with the module that owns them.

## Module shape

Start each module with the smallest useful structure. A module may evolve toward:

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

Create directories when they have a concrete responsibility and content.

A module should remain understandable through its capability vocabulary rather than through generic architectural layer names.

## Public module boundary

Each module exposes a small documented public API.

Consumers should depend on:

- public data types;
- public functions or classes;
- explicit protocols for real variation points;
- stable module entry points.

Keep implementation details local to the producer module, including internal model objects, caches, training structures, backend-specific types, storage layout, and third-party runtime objects.

A useful dependency shape is:

```text
consumer -> producer public API -> producer implementation
```

## SOLID rules

### Single Responsibility Principle

Give every module, class, function, and service one coherent reason to change.

Prefer focused components such as:

```text
TemporalFusion
ObservationMatcher
PointEncoder
PoseEstimator
SemanticMapReader
SemanticMapWriter
```

Split behavior when responsibilities evolve independently.

### Open/Closed Principle

Represent genuine variation points with stable contracts.

Typical examples in this project include:

- multiple pose estimators;
- multiple fusion strategies;
- multiple point encoders;
- multiple persistence backends;
- research variants compared by experiments.

Consumers should remain stable while implementations vary behind the same public behavior.

### Liskov Substitution Principle

Implementations of the same public contract must be safely interchangeable.

Document and test substitution-relevant invariants such as:

- units;
- coordinate frames;
- timestamp semantics;
- ordering guarantees;
- tensor or embedding dimensions;
- lifecycle behavior;
- error behavior.

### Interface Segregation Principle

Shape interfaces around one consumer need.

For example, map reading and map writing can be separate contracts when consumers require only one side.

Keep interfaces narrow enough that an implementation can satisfy them without unrelated responsibilities.

### Dependency Inversion Principle

High-level orchestration depends on stable capability behavior.

For example:

```text
mapping runtime -> PoseEstimator <- concrete LiDAR-inertial estimator
```

The runtime depends on the capability it needs. The concrete estimator supplies that capability.

Use this pattern at boundaries that are replaceable, expensive, external, or intentionally varied by research experiments.

## When to create an interface or protocol

Create one when a stable behavior boundary is useful because:

- multiple implementations exist;
- an implementation is intentionally replaceable;
- a research experiment compares implementations;
- a third-party dependency should stay contained;
- a module boundary needs a stable contract;
- tests need a lightweight substitute for an expensive or external component.

Keep direct construction and concrete local code for behavior with no meaningful variation point.

## Shared types and contracts

Place a type in repository-wide shared code when both conditions are true:

1. the concept is owned by the system as a whole;
2. multiple modules must agree on exactly the same stable meaning.

Good candidates:

```text
Timestamp
FrameId
Pose
RigidTransform
MapId
ArtifactId
Provenance
```

Capability-specific types remain with their capability. A learned point representation, for example, is owned by `point-representation` even when another module consumes it.

## External integrations

Keep each external integration close to its owning capability.

Examples:

```text
modules/state-estimation/
    concrete LiDAR-inertial odometry integration

modules/geometric-map/
    point-cloud or reconstruction backend used by geometric mapping

modules/visual-perception/
    model runtime used by visual perception
```

Create repository-wide integration infrastructure only for behavior genuinely shared by unrelated capabilities with the same semantics.

## Applications

`apps/` owns runnable composition.

Applications may:

- select concrete implementations;
- connect module inputs and outputs;
- configure workflows;
- expose CLI, GUI, TUI, or service entry points;
- load and persist composed map state;
- coordinate runtime lifecycle.

Research algorithms remain owned by the module that implements the capability.

## Research implementations

Design public architecture around project responsibilities.

A paper, model, framework, dataset, or external repository may inform a concrete implementation. Record that scientific provenance in the relevant module documentation while keeping public names capability-based.

For example:

```text
public capability: PoseEstimator
concrete implementation: FAST_LIO-backed estimator
```

Issues should describe the capability, observable behavior, tests, acceptance criteria, and relevant constraints. Research references belong in module documentation when they help explain the implementation or comparison.

## Explicit multimodal data flow

Prefer visible transformations with explicit project types:

```text
observation
    -> boundary validation
    -> capability transformation
    -> explicit output
    -> next capability
```

Preserve metadata required to interpret or reproduce results. Depending on the boundary, this includes:

- timestamp;
- sensor identity;
- coordinate frame;
- units;
- calibration or transform provenance;
- source observation identity;
- confidence or uncertainty;
- model/checkpoint provenance.

Validate these properties when data crosses a module boundary.

## Configuration

Keep algorithm configuration with the module that owns the algorithm.

Keep composition configuration with the application or experiment that selects and connects implementations.

Prefer typed, explicit configuration with reproducible defaults for parameters that affect experiments.

## Error handling

Validate boundary invariants early and return actionable errors.

Important validation targets include:

- coordinate-frame compatibility;
- timestamp synchronization;
- transform validity;
- tensor and embedding dimensions;
- point attributes;
- map/version compatibility;
- required provenance.

## Testing expectations

Use the narrowest test that protects the required behavior:

- unit tests for deterministic local behavior;
- contract tests for interchangeable implementations;
- integration tests for module boundaries;
- representative end-to-end tests for application composition;
- benchmarks for runtime, memory, and accuracy-sensitive research components;
- regression tests for previously observed failure modes.

Test observable behavior and boundary guarantees so internal implementations can evolve safely.

## Naming

Name components by their capability and responsibility.

Prefer:

```text
PointEncoder
PoseEstimator
ObservationMatcher
TemporalFusion
SemanticMapReader
SemanticMapWriter
```

Use generic names such as `Manager`, `Helper`, `Utils`, `Processor`, or `Service` only when they are genuinely the clearest domain term.

## Documentation

Repository-wide architecture and engineering decisions belong in `docs/`.

Module behavior, algorithms, model choices, implementation rationale, benchmarks, limitations, and research references belong in `modules/<module>/docs/`.

Update documentation in the same change when a boundary, ownership rule, public contract, or dependency direction changes.

## Decision test before adding complexity

Before adding a layer, global package, registry, factory, base class, or interface, answer:

1. What concrete responsibility does it represent?
2. Which dependency or variation point does it make explicit?
3. Which consumer becomes simpler or safer because of it?
4. Which test or replacement scenario benefits from it?
5. Is this the smallest structure that communicates the intent clearly?

Use the abstraction when these answers are concrete. Otherwise keep the implementation local and direct.

## Source of truth

For repository architecture and engineering principles, also read:

- `docs/architecture.md`
- `docs/engineering-principles.md`
- `docs/documentation-policy.md`
- `modules/README.md`

When code and documentation diverge, treat the inconsistency as part of the change and restore one clear architectural source of truth.