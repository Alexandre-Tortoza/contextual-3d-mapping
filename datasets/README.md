# Datasets

This directory stores dataset metadata required for reproducible experiments, not large raw datasets.

```text
datasets/
├── manifests/
├── schemas/
└── splits/
```

Use it for dataset manifests, expected sensor availability, sequence definitions, calibration metadata references, schema descriptions, evaluation splits, and reproducibility metadata.

Large images, point clouds, bags, archives, and generated map artifacts should remain outside Git and be referenced through manifests or configured storage adapters.
