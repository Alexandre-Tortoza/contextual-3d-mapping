from __future__ import annotations

import json
from pathlib import Path

import pytest

import image_context.cli as cli
from image_context.cli import _build_parser
from image_context.models import ImageSample


def test_sample_parses_arguments() -> None:
    args = _build_parser().parse_args(
        ["sample", "--config", "custom.yaml", "--sample-size", "5", "--seed", "7"]
    )

    assert args.command == "sample"
    assert args.config == Path("custom.yaml")
    assert args.sample_size == 5
    assert args.seed == 7


def test_main_extracts_and_records_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bag_path = tmp_path / "corridor.bag"
    bag_path.touch()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"dataset:\n"
        f"  bag_path: {bag_path}\n"
        f"  image_topic: /camera_1/image_raw\n"
        f"  sample_size: 2\n"
        f"  seed: 42\n"
        f"output_directory: {tmp_path / 'runs'}\n"
        f"run_id: test-run\n",
        encoding="utf-8",
    )

    def fake_sample(
        self: object, count: int, seed: int, destination: Path
    ) -> tuple[ImageSample, ...]:
        samples = []
        for index in range(count):
            frame_directory = destination / f"frame-{index:06d}"
            frame_directory.mkdir(parents=True, exist_ok=True)
            image_path = frame_directory / "image.png"
            image_path.write_bytes(b"")
            samples.append(
                ImageSample(
                    frame_id=f"frame-{index:06d}",
                    source_index=index,
                    timestamp_ns=index,
                    width=10,
                    height=10,
                    image_path=image_path,
                )
            )
        return tuple(samples)

    monkeypatch.setattr(cli.RosbagImageSampler, "sample", fake_sample)

    exit_code = cli.main(["sample", "--config", str(config_path)])

    assert exit_code == 0
    selection_path = tmp_path / "runs" / "test-run" / "selected_frames.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    assert len(selection["frames"]) == 2
