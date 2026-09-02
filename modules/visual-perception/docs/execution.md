# Execution

## Model lifecycle (issue #171)

`application/lifecycle.ModelLifecycleManager` creates one heavyweight adapter per stage,
runs it, records `StageMetrics` (load time, inference time, peak memory), and releases
the reference — so the canonical pipeline never needs every model resident at once.

```python
manager = ModelLifecycleManager()
with manager.stage("region_discovery", lambda: load_real_backend(config)) as model:
    result = model.discover(image, config)
```

Peak memory is measured as CPU-RSS in this GPU-free environment (a proxy, not real VRAM);
real peak-VRAM measurement is part of the real-hardware validation in #190. An
out-of-memory condition during load or inference surfaces as
`domain.errors.BackendExecutionError` — never a silent substitution of a different
backend or configuration.

## Stage cache (issue #170)

`application/cache.StageCache` persists one JSON record per completed stage, keyed by a
chained fingerprint (`compute_fingerprint`): a stage's fingerprint hashes its own version
and configuration together with every upstream stage's fingerprint. Consequences:

- an identical run reuses every valid cached stage;
- changing one stage's configuration invalidates that stage and everything computed from
  its output (its downstream dependents) — sibling stages that do not depend on it stay
  valid;
- an interrupted run is resumable: the next run with the same fingerprints picks up
  where it left off;
- `CACHE_SCHEMA_VERSION` rejects a cache entry written by an incompatible module version
  instead of reusing it.

## Quality checks

```bash
cd modules/visual-perception
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

`contextual_mapping_contracts` (and, for the integration tests only,
`contextual_mapping_adapters`/`contextual_mapping_datasets`) resolve from their source
trees via `[tool.pytest.ini_options].pythonpath` in `pyproject.toml` until those packages
have their own installable build (see [model-backends.md](model-backends.md) for the
equivalent gap on the ML side).
