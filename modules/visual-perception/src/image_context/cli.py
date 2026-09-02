"""Command-line entry point for dataset sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from image_context.adapters.rosbag_sampler import RosbagImageSampler
from image_context.artifacts import ArtifactRepository
from image_context.config import PipelineConfig, load_config


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        config = _apply_overrides(load_config(args.config), args)
        artifacts = _initialize_artifacts(config, overwrite=args.overwrite)
        sampler = RosbagImageSampler(config.dataset.bag_path, config.dataset.image_topic)
        samples = sampler.sample(
            config.dataset.sample_size, config.dataset.seed, artifacts.frames_directory
        )
        artifacts.write_selection(samples)
        print(f"Extracted {len(samples)} images to {artifacts.frames_directory}")
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-context",
        description="Sample ROS bag images for the upcoming image-context module.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    sample = subparsers.add_parser("sample")
    sample.add_argument("--config", type=Path, default=Path("config.yaml"))
    sample.add_argument("--sample-size", type=int)
    sample.add_argument("--seed", type=int)
    sample.add_argument("--output", type=Path)
    sample.add_argument("--run-id")
    sample.add_argument("--overwrite", action="store_true")
    return parser


def _apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    dataset = replace(
        config.dataset,
        sample_size=(
            config.dataset.sample_size if args.sample_size is None else args.sample_size
        ),
        seed=config.dataset.seed if args.seed is None else args.seed,
    )
    output = config.output_directory if args.output is None else args.output.resolve()
    run_id = config.run_id if args.run_id is None else args.run_id
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run-id must be a single non-empty directory name.")
    return replace(config, dataset=dataset, output_directory=output, run_id=run_id)


def _initialize_artifacts(config: PipelineConfig, *, overwrite: bool) -> ArtifactRepository:
    configuration = {
        "dataset": {
            "bag_path": str(config.dataset.bag_path),
            "image_topic": config.dataset.image_topic,
            "sample_size": config.dataset.sample_size,
            "seed": config.dataset.seed,
        },
    }
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    artifacts = ArtifactRepository(config.output_directory / config.run_id)
    artifacts.initialize(
        {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "configuration": configuration,
        },
        overwrite=overwrite,
    )
    return artifacts


if __name__ == "__main__":
    raise SystemExit(main())
