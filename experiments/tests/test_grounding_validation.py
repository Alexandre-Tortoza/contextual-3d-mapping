"""Regressão de identidade e suporte nas métricas da ablação espacial."""

from visual_perception_experiments.grounding_validation import _metrics


# IDs locais dos diagnostics não podem sobrescrever a identidade composta,
# porque isso confundiria regiões de frames diferentes no relatório multi-view.
def test_metrics_preserve_frame_identity_and_do_not_invent_agreement() -> None:
    """Mede pontos, tentativas e identidade sem inferir concordância de uma view."""
    payload = {
        "points": [{"context": {"label": "object", "spatial_support": 0.7,
                                "support_state": "uncorroborated"}}],
        "observations": [{"semantic_association_counts": {"interior": 1, "boundary": 2},
                          "semantic_label_association_counts": {"object": {"interior": 1, "boundary": 2}}}],
        "regions": [{"region_id": "frame-a:region-a", "observation_frame_id": "frame-a", "label": "object",
                     "spatial_diagnostics": {"region_id": "region-a", "semantic_area": 50}}],
        "context_summary": {},
    }
    result = _metrics(payload)
    assert result["regions"][0]["region_id"] == "frame-a:region-a"
    assert result["labels"]["object"]["multi_view_agreement"] is None
    assert result["labels"]["object"]["spatial_support_mean"] == 0.7
    assert result["safe_interior_association_count"] == 1
    assert result["rejected_strong_boundary_association_count"] == 2
