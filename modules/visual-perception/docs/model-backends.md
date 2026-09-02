# Model Backends

## Current status: everything is a GPU-free fake

Every canonical stage defaults to `backend="fake"` (`config.py`) and runs against a
deterministic, model-free implementation under `infrastructure/fakes/`:

| Port | Fake | Real adapter (blocked) |
| --- | --- | --- |
| `RegionDiscoverer` | `fake_region_discoverer.FakeRegionDiscoverer` | `adapters/region_discovery_backend.py` (#186) |
| `DenseFeatureExtractor` | `fake_feature_extractor.FakeDenseFeatureExtractor` | `adapters/feature_extraction_backend.py` (#187) |
| `LanguageAlignedEncoder` | `fake_language_encoder.FakeLanguageAlignedEncoder` | `adapters/language_embedding_backend.py` (#188) |
| `MultimodalReasoner` | `fake_multimodal_reasoner.FakeMultimodalReasoner` | `adapters/multimodal_reasoning_backend.py` (#189) |

The fakes are real, content-sensitive implementations (connected-component region
discovery, average-pooled feature grids, seeded deterministic embeddings, brightness/color
heuristics for scene and region responses) — not stubs — so every stage, and the
canonical pipeline end-to-end, is genuinely exercised by tests without a GPU.

## Why the real adapters are stubs right now

This development environment has no GPU (`nvidia-smi` fails, no `torch` installed). The
four `infrastructure/adapters/*_backend.py` classes satisfy their port's shape but raise
`domain.errors.BackendUnavailableError` when called, so a caller that accidentally wires
one in fails explicitly and immediately rather than silently getting fake output.

Issues #174 (benchmark-driven backend selection), #186-#189 (real adapters), and #190
(reference-hardware validation) stay **open** until this module is finished on a
GPU-equipped machine. `benchmarks/backend_benchmark.py` and
`application/execution_profile.py` already implement the selection *harness and policy*
(quality-first subject to the memory budget, latency never excludes a candidate that
fits) — what is missing is real candidates to run it against.

## What "done" looks like for #174/#186-#190

1. Implement one real adapter per port, translating canonical inputs/outputs and
   surfacing backend failures as `BackendExecutionError` (never a backend-specific
   exception or tensor type).
2. Run `benchmarks/backend_benchmark.benchmark_candidate` for each candidate under the
   reference GPU budget; record latency, peak VRAM, and a task-relevant quality score.
3. Use `application/execution_profile.select_research_quality_backend` to pick the
   highest-quality candidate that fits the memory budget, and
   `additional_compute_is_justified` before enabling multi-scale tiling in the reference
   config.
4. Run `application/pipeline.run_canonical_pipeline` end-to-end with the real adapters on
   a representative image set (#190), verifying no OOM, valid stage-cache reuse on a
   repeated run, and that outputs still pass the quality auditor.
5. Update this file's status table and close #174/#186-#190.
