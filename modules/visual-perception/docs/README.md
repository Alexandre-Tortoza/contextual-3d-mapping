# Visual Perception Documentation

This folder documents `visual-perception`'s architecture, execution flow, artifacts, model
backends, and research traceability.

## Contents

- [architecture.md](architecture.md) — module responsibility, layers, the canonical
  pipeline diagram, and the image-coordinate invariants.
- [pipelines.md](pipelines.md) — what each canonical stage does, and where the removed
  legacy baselines are preserved for comparison.
- [execution.md](execution.md) — model lifecycle, memory diagnostics, and the stage
  cache.
- [artifacts.md](artifacts.md) — the canonical serialization format and what a persisted
  run looks like.
- [model-backends.md](model-backends.md) — current backend status (all GPU-free fakes),
  what is blocked on real hardware, and how selection will work once benchmarked.
- [research-traceability.md](research-traceability.md) — which mechanisms are adopted
  from prior work, which are project-specific engineering, and what remains a future
  idea.

## Status

Every canonical stage (region discovery, tiling/merge, dense features, mask-aware
pooling, language-aligned embedding, scene/region semantic interpretation, relation
generation, quality audit) is implemented and covered by unit and end-to-end tests using
GPU-free fakes. Real model backends (#186-#189) and hardware validation (#174, #190) are
not implemented yet: this development environment has no GPU. See
[model-backends.md](model-backends.md).
