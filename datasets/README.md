# Datasets

This directory owns the repository conventions for local dataset files and the metadata required for reproducible experiments.

```text
datasets/
├── raw/
│   └── <dataset-name>/
│       └── ...
├── manifests/
├── schemas/
├── splits/
└── contextual_mapping_datasets/
```

## Raw dataset layout

Downloaded or extracted dataset files must live under:

```text
datasets/raw/<dataset-name>/*
```

Each dataset gets its own directory and should preserve the upstream dataset layout whenever practical. Dataset adapters should read from this dataset root instead of scattering source files across the repository.

Examples:

```text
datasets/raw/cerberus-subt/...
datasets/raw/grandtour/...
datasets/raw/tartanground/...
```

`datasets/raw/` is local working data. Its contents are intentionally excluded from Git because RGB images, LiDAR point clouds, ROS bags, archives, and similar source artifacts can be very large.

## Versioned dataset support

Use the remaining directories for repository-tracked information:

- `manifests/` describes dataset identity, sequences, sensor sources, clocks, frames, calibration references, provenance, and local source locations.
- `schemas/` contains dataset-related schemas and validation definitions.
- `splits/` contains reproducible train, validation, test, and evaluation splits.
- `contextual_mapping_datasets/` provides the versioned manifest model used by dataset adapters and experiments.

The current manifest schema version `1.0` describes dataset and sequence identity, external sensor sources, clocks, frames, calibration references, and optional evaluation split membership.

Generated maps, model checkpoints, caches, and experiment outputs are not raw dataset files and should remain in their owning runtime, experiment, or configured artifact storage location.
