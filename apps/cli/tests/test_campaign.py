"""Testes da seleção declarativa das campanhas de trechos."""

from pathlib import Path

import pytest

from contextual_mapping_cli.campaign import load_campaign

CAMPAIGNS = Path(__file__).resolve().parents[1] / "configs" / "campaigns"


# Protege a cardinalidade escolhida para o experimento antes de acionar GPU ou
# rosbag: uma alteração acidental na configuração deve falhar cedo.
def test_corridor_campaign_has_ten_segments_and_four_frame_positions() -> None:
    """Valida os dez cenários, a divisão indoor/outdoor e os quatro offsets."""
    campaign = load_campaign(CAMPAIGNS / "corridor-02-context-campaign.toml")
    assert len(campaign.segments) == 10
    assert sum(item.environment == "indoor" for item in campaign.segments) == 8
    assert sum(item.environment == "outdoor" for item in campaign.segments) == 2
    assert campaign.keyframe_offsets_s == (0.0, 0.25, 0.5, 1.0)
    assert campaign.voxel_size_m == 0.05
    assert campaign.max_viewer_points == 150_000
    assert campaign.reasoning_backend is None


# A campanha do Gemini usa trechos novos, a 2 fps, e declara o backend remoto.
def test_gemini_campaign_uses_new_segments_at_two_fps() -> None:
    """Valida dois indoor e um outdoor fora dos trechos da campanha 10x4fps."""
    gemini = load_campaign(CAMPAIGNS / "corridor-02-context-gemini-2fps.toml")
    reference = load_campaign(CAMPAIGNS / "corridor-02-context-campaign.toml")
    assert gemini.reasoning_backend == "gemini_robotics_er"
    assert gemini.keyframe_offsets_s == (0.0, 0.5)
    assert [item.environment for item in gemini.segments] == ["indoor", "outdoor", "indoor"]
    assert {item.start_s for item in gemini.segments}.isdisjoint({item.start_s for item in reference.segments})


# Combinações ambíguas falham antes de abrir a rosbag.
@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("keyframe_offsets_s = [0.5, 0.0]", "offsets"),
        ("keyframe_offsets_s = [0.0, 2.0]", "offsets"),
        ('reasoning_backend = "gpt"', "reasoning_backend"),
    ],
)
def test_invalid_campaigns_are_rejected(tmp_path: Path, body: str, message: str) -> None:
    """Recusa offsets fora de ordem ou da janela e backend desconhecido."""
    config = tmp_path / "campaign.toml"
    config.write_text(
        f"""campaign_id = "x"\nduration_s = 1.0\nvoxel_size_m = 0.05\nmax_viewer_points = 10\n{body}\n"""
        + ("keyframe_offsets_s = [0.0]\n" if "keyframe_offsets_s" not in body else "")
        + """[[segments]]\nid = "a"\nenvironment = "indoor"\nstart_s = 1.0\n""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        load_campaign(config)
