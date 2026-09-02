# Documentation Policy

The repository uses two documentation levels to keep architectural context separate from module implementation details.

## Repository-level documentation

Location:

```text
docs/
```

This level owns documentation that applies to the project as a whole:

- repository architecture and directory conventions;
- module topology and high-level flow;
- cross-module dependency rules;
- integration and contract conventions;
- application composition rules;
- repository-wide testing, evaluation, and experimentation conventions;
- project-level decisions that affect more than one module.

Repository-level documentation may name modules and show how they connect, but should not explain their internal algorithms or implementation choices.

## Module-level documentation

Location:

```text
modules/<module>/docs/
```

Each module will own its detailed technical documentation when development begins. This may include:

- module goals and responsibilities;
- public input and output contracts;
- internal architecture;
- algorithms and models;
- preprocessing and postprocessing;
- training and inference workflows;
- configuration;
- datasets used specifically by the module;
- benchmarks and evaluation methodology;
- implementation rationale;
- limitations and known failure modes;
- research references relevant to that implementation.

Each module should eventually expose an entry point such as:

```text
modules/<module>/docs/index.md
```

## Separation rule

If a document answers **how the system is composed**, it belongs in root `docs/`.

If a document answers **how one module works**, it belongs in that module's `docs/`.

The root documentation should link to module documentation when needed instead of duplicating it. This keeps independently evolving research implementations from making the global architecture documentation unstable.
