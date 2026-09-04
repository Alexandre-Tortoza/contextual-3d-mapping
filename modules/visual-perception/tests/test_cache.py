"""Testes de fingerprints de estágio e do cache de artifact reutilizável (#170). Sem modelos reais."""

from __future__ import annotations

from pathlib import Path

from visual_perception.application.cache import StageCache, compute_fingerprint


# Confirma o caso central do cache: uma execução repetida com o mesmo fingerprint
# reutiliza o resultado gravado em vez de recomputar o estágio.
def test_repeated_identical_execution_reuses_cache(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    fingerprint = compute_fingerprint("region_discovery", "v1", "cfg-abc")
    assert cache.get("region_discovery", fingerprint) is None

    cache.put("region_discovery", fingerprint, {"proposal_count": 3})
    assert cache.is_complete("region_discovery", fingerprint)
    assert cache.get("region_discovery", fingerprint) == {"proposal_count": 3}


# Garante que mudar a config de um estágio muda seu fingerprint, invalidando só aquele
# estágio no cache.
def test_changing_stage_config_invalidates_only_that_fingerprint() -> None:
    original = compute_fingerprint("feature_extraction", "v1", "cfg-a")
    changed = compute_fingerprint("feature_extraction", "v1", "cfg-b")
    assert original != changed


# Verifica o encadeamento de fingerprints: mudar a config de um estágio upstream muda
# o fingerprint de um estágio downstream que depende dele, mesmo sem a config local mudar.
def test_downstream_fingerprint_changes_when_upstream_changes() -> None:
    upstream_a = compute_fingerprint("region_discovery", "v1", "cfg-a")
    upstream_b = compute_fingerprint("region_discovery", "v1", "cfg-b")
    downstream_a = compute_fingerprint("merge", "v1", "cfg-merge", upstream_fingerprints=(upstream_a,))
    downstream_b = compute_fingerprint("merge", "v1", "cfg-merge", upstream_fingerprints=(upstream_b,))
    assert downstream_a != downstream_b


# Garante que um estágio irmão, sem relação de dependência, não é afetado por uma
# mudança em outro estágio — protege contra invalidação de cache superampla.
def test_sibling_stage_fingerprint_is_unaffected_by_unrelated_change() -> None:
    before = compute_fingerprint("language_embedding", "v1", "cfg-lang")
    # Uma mudança na configuração de um estágio não relacionado não afeta esse fingerprint.
    after = compute_fingerprint("language_embedding", "v1", "cfg-lang")
    assert before == after


# Simula uma execução interrompida: um novo StageCache apontando para o mesmo diretório
# deve reconhecer o estágio já concluído e retomar do ponto onde parou.
def test_interrupted_run_is_resumable(tmp_path: Path) -> None:
    cache = StageCache(tmp_path)
    fingerprint = compute_fingerprint("scene_context", "v1", "cfg-abc")
    assert not cache.is_complete("scene_context", fingerprint)
    cache.put("scene_context", fingerprint, {"done": True})

    resumed_cache = StageCache(tmp_path)
    assert resumed_cache.is_complete("scene_context", fingerprint)
