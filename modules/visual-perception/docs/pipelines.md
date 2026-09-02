# Pipelines

## Canonical pipeline

`application/pipeline.run_canonical_pipeline` (#169) is the module's single production
entry point. Each stage is independently testable and independently replaceable behind
its own port:

| Stage | Module | Issue |
| --- | --- | --- |
| Multi-scale tiling & remapping | `application/tiling.py` | #159 |
| Region discovery | `ports/region_discovery.py` + `application/pipeline.py` | #158 |
| Cross-scale region merge | `application/region_merge.py` | #160 |
| Dense visual features | `ports/feature_extraction.py` | #161 |
| Mask-aware region pooling | `application/pooling.py` | #162 |
| Language-aligned embedding | `application/language_embedding.py` | #163 |
| Scene-level context | `application/scene_context.py` | #164 |
| Region-level semantics | `application/region_semantics.py` | #165 |
| Relation generation | `application/relation_generation.py` | #167 |
| Quality audit | `application/quality_audit.py` | #168 |

Optional, non-canonical capabilities compose on top of the pipeline output rather than
inside it:

- `application/fusion.py` (#182) — multi-source region/claim fusion;
- `application/refinement.py` (#183) — uncertainty-driven selective reprocessing;
- `application/execution_profile.py` (#181) — the quality-first backend selection policy.

## Legacy baselines

The former standalone `image-context` laboratory's two production pipelines —
three-pass VLM (Grounding DINO -> SAM2) and region-first (SAM2 -> DINOv2 -> Qwen) — were
removed before this migration (see the former repository's issue #4). They remain
recoverable from that repository's git history for scientific comparison, but are not
part of this module's canonical path: `run_canonical_pipeline` never exposes a legacy
strategy selector. See `benchmarks/legacy/README.md` (#176) for exactly what is and is
not reproduced here.

## Dataset/ROS bag sampling

Sampling images from a ROS bag or dataset is not part of this module's public
capability (#151): visual-perception consumes normalized RGB observations from
`[adapters]` (`contextual_mapping_adapters.CanonicalObservation`, kind `"rgb"`) through
`infrastructure/integration/rgb_adapter_boundary.py` (#177). The original
`image_context.adapters.rosbag_sampler` and its `sample` CLI command were removed during
migration rather than kept as dead weight in this module's public surface; that
capability's home is a dataset/transport adapter, not visual-perception.
