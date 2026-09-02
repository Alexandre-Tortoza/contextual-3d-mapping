# Visual Perception

`visual-perception` turns one canonical RGB image observation into a structured,
auditable visual observation: discovered regions with masks and boxes, dense and
language-aligned region embeddings, scene- and region-level semantic claims, candidate
image-level relations, and a quality audit — with model and configuration provenance
attached throughout.

> Detailed documentation: [`docs/`](docs/README.md).

The module is independently testable and every canonical stage is replaceable behind a
port. It ships with complete, deterministic GPU-free fakes for every backend so its
contracts, pipeline, cache, and integration boundaries can be fully exercised without a
GPU or a model download; real backends are tracked separately (see
[docs/model-backends.md](docs/model-backends.md)).

## Responsibilities

- consume canonical RGB observations emitted by `[adapters]` (not read datasets/ROS bags
  directly);
- discover, tile, and merge image regions into stable canonical regions;
- pool dense visual features and produce language-aligned embeddings per region;
- interpret scene- and region-level semantics as auditable claims, not single labels;
- generate candidate 2D relations between regions;
- audit the resulting observation for structural consistency and contradictions;
- cache expensive stages and serialize the canonical observation for persistence.

## Non-responsibilities

- dataset/ROS bag sampling and transport (`[adapters]`);
- calibration, cross-sensor projection, LiDAR association (`sensor-association`);
- persistent geometric/semantic map construction, scene graphs.

## Structure

```text
visual-perception/
├── README.md
├── docs/
├── benchmarks/
├── src/
│   └── visual_perception/
│       ├── domain/
│       ├── ports/
│       ├── application/
│       ├── infrastructure/
│       │   ├── fakes/
│       │   ├── adapters/
│       │   └── integration/
│       └── config.py
└── tests/
```

## Development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

`contextual_mapping_contracts` (and, for integration tests only,
`contextual_mapping_adapters`/`contextual_mapping_datasets`) resolve from their source
trees via `pyproject.toml`'s pytest `pythonpath` until those packages have their own
installable build.
