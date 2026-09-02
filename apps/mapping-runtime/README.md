# Mapping Runtime

`mapping-runtime` is the composition root for building or updating maps from live sensors, recorded sessions, or dataset adapters.

It owns configuration, dependency wiring, lifecycle, and execution order. It does not implement research capabilities that belong to modules.

Initial structure:

```text
mapping-runtime/
├── README.md
├── configs/
└── src/
```

The runtime should consume public contracts and module entry points so the same downstream pipeline can operate with live sensors, recorded data, datasets, simulator output, or test fixtures.
