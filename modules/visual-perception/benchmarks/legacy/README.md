# Legacy baselines

Issue: #176.

The two historical production pipelines from the standalone `image-context`
laboratory —

- **baseline**: three-pass VLM -> Grounding DINO -> SAM2;
- **region-first**: SAM2 automatic -> DINOv2 -> Qwen global/local;

were removed from the codebase before the migration into this module (see
`docs/architecture.md`, which documents their removal and links back to the
original `image-context` issue history). They are not present here as
executable code.

## Why they are not reproduced as running code

Reintroducing them would require the original heavyweight model stack
(Grounding DINO, SAM2, DINOv2, Qwen) and their original prompts/configs,
none of which are available in this environment. Recreating them from
scratch without the original artifacts would not be a faithful scientific
baseline, so this module does not fabricate a substitute.

## What is preserved instead

- Both pipelines remain fully recoverable from git history in the original
  `image-context` repository (see the commit referenced by
  `docs/architecture.md`).
- The canonical pipeline (`application/pipeline.py`, #169) does not expose
  any legacy strategy selector: there is exactly one production entry point,
  satisfying #176's requirement that canonical use never require choosing a
  legacy variant.
- If a future scientific comparison needs to re-run a legacy baseline, its
  harness would live under this directory, isolated from
  `application/pipeline.py` and using its own run-artifact namespace so it
  cannot collide with canonical run artifacts (`benchmarks/harness.py`'s
  `DatasetReference`/`BenchmarkReport` types are reusable for that
  comparison once the legacy code is reintroduced).
