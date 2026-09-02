# Dataset Guidance

These rules apply to work under `datasets/`.

## Canonical raw data location

All downloaded or extracted source dataset files belong under:

```text
datasets/raw/<dataset-name>/*
```

Use one stable directory name per dataset. Preserve the upstream dataset layout inside that directory whenever practical.

Dataset adapters, manifests, experiments, and development scripts should treat `datasets/raw/<dataset-name>/` as the canonical local root for that dataset rather than introducing dataset files elsewhere in the repository.

## Git policy

`datasets/raw/` contains local working data and is ignored by Git. Do not commit dataset images, point clouds, ROS bags, archives, or other large source artifacts.

Repository-tracked dataset information belongs in:

- `manifests/` for identity, sensor sources, sequences, calibration references, provenance, and source locations;
- `schemas/` for validation and interchange definitions;
- `splits/` for reproducible train, validation, test, and evaluation splits;
- `contextual_mapping_datasets/` for dataset manifest code shared by adapters and experiments.

Do not place generated maps, caches, checkpoints, or experiment outputs in `raw/`; those artifacts belong to the component or experiment that produces them.
