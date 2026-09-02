# Research Traceability

This page distinguishes what this module adopts from established techniques, what is
project-specific engineering, and what is still a future idea — so a reader can tell
which design decisions are backed by prior work versus this repository's own choices.

## Adopted mechanisms (implementation-agnostic boundary, real backend pending)

The module's *boundaries* are shaped around well-established technique families, even
though no real backend is wired in yet (see [model-backends.md](model-backends.md)):

- **Region discovery** (`ports/region_discovery.py`, #158): the boundary is shaped to fit
  class-agnostic/promptable segmentation models in the SAM family, which is why a
  proposal carries a mask, a box, and a *geometric* confidence only — no semantic label.
- **Dense visual features + mask-aware pooling** (`ports/feature_extraction.py`,
  `application/pooling.py`, #161-#162): shaped for spatial feature grids from
  self-supervised vision backbones such as DINOv2, where a region's embedding is pooled
  from the grid rather than re-encoded from a crop.
- **Language-aligned embedding** (`ports/language_embedding.py`, #163): shaped for
  CLIP-style contrastive image/text encoders, kept as a *second*, independent
  representation from the dense visual embedding above (different space, different
  lifecycle).
- **Multimodal reasoning** (`ports/multimodal_reasoning.py`, #164-#165, #189): shaped for
  a vision-language model prompted separately for scene-level and region-level
  structured output, mirroring how such models are typically used for grounded
  captioning/VQA-style tasks.

None of these are cited as "the" chosen model: which concrete checkpoint satisfies each
boundary is an open, benchmark-driven decision (#174), not an architectural one.

## Project-specific engineering contributions

These are this repository's own design choices, not taken from a specific paper:

- **Chained stage fingerprints for caching** (`application/cache.py`, #170): fingerprints
  compose upstream fingerprints so changing one stage's configuration invalidates exactly
  that stage and its dependents, not the whole run.
- **Claims-not-labels semantic representation** (`domain/semantics.py`, #156):
  representing interpretation as a set of auditable, possibly-contradictory claims with
  per-claim confidence and evidence, rather than collapsing each region to one label and
  one score.
- **Deterministic geometric relation derivation** (`application/relation_generation.py`,
  #167): overlap/containment/adjacency relations derived purely from mask geometry
  (mutual containment ratio, IoU, bounding-box gap), kept explicitly distinguishable from
  model-inferred relations.
- **Two-tier mask-aware pooling** (`application/pooling.py`, #162): a coarse
  cell-center-inclusion baseline plus a per-pixel nearest-cell "high-resolution" path
  specifically to keep sub-grid-cell regions representable — an engineering response to
  the small-region alignment problem, not a published pooling method.
- **Calibrated multi-source fusion via reused merge logic**
  (`application/fusion.py`, #182): treating a second perception source as just another
  contributor to the existing cross-scale merge, rather than a separate fusion algorithm.

## Future ideas (not implemented)

- Real quantitative backend selection and quality/ablation benchmarking (#174, #175) —
  requires GPU hardware this environment does not have.
- Uncertainty-driven selective refinement (`application/refinement.py`, #183) is
  implemented and unit-tested against fakes, but has never been evaluated against a real
  backend's actual uncertainty distribution — whether its trigger thresholds are
  well-calibrated is an open question for #175's ablation harness.
- Preserving the former standalone repository's two legacy pipelines as *runnable*
  comparison baselines (#176) — currently only documented as recoverable from that
  repository's git history; see `benchmarks/legacy/README.md`.
