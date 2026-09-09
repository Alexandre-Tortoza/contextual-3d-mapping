"""Compara dois runs versionados nos eixos estruturais que o módulo mede.

Existe porque "melhorou" não é uma afirmação verificável sem uma tabela lado a
lado, e porque a comparação precisa ser refeita do mesmo jeito toda vez. Ela lê
apenas ``manifest.json`` e ``diagnostics.json`` dos dois runs: nenhum modelo é
reexecutado, e nenhum número é recalculado a partir das observações — o que é
comparado é exatamente o que cada run registrou sobre si mesmo.

**Nenhuma métrica de acurácia é produzida aqui.** Enquanto as #210/#211
estiverem abertas, o conjunto de referência não tem anotação humana revisada, e
toda afirmação de correção semântica seria a saída do próprio pipeline promovida
a ground truth. O que esta ferramenta compara é estrutura, cobertura e custo.

Uso, a partir de ``modules/visual-perception``:

    python benchmarks/compare_runs.py \\
      --baseline benchmarks/results/samples/<antigo> \\
      --candidate benchmarks/results/samples/<novo>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: Frames vinculantes do projeto, na ordem canônica. Comparar outro conjunto é
#: possível por ``--frame-id``, mas o default é este de propósito: trocar os
#: frames entre duas comparações invalidaria as duas.
BINDING_FRAMES = ("corridor-02-000", "corridor-02-008", "corridor-02-017")

#: Eixos comparados, na ordem em que a tabela os apresenta. Cada entrada é
#: ``(rótulo, caminho no diagnostics/manifest, formato)``. A lista é explícita
#: para que um eixo novo apareça de propósito, e não por um campo ter surgido.
_AXES: tuple[tuple[str, str, str], ...] = (
    ("proposals", "diag:proposal_count", "{}"),
    ("regiões canônicas", "diag:region_count", "{}"),
    ("labels crus distintos", "diag:mode_collapse.distinct_labels", "{}"),
    ("conceitos canônicos", "diag:contextual.distinct_canonical_concepts", "{}"),
    ("label dominante", "diag:mode_collapse.dominant_label", "{}"),
    ("fração dominante", "diag:mode_collapse.dominant_fraction", "{:.2f}"),
    ("scene echo (1ª hipótese)", "diag:scene_echo_label_count", "{}"),
    ("scene echo (qualquer)", "diag:contextual.scene_echo_any_assertion", "{}"),
    ("confiança degenerada", "diag:semantic_confidence.degenerate", "{}"),
    ("claims não pontuadas", "diag:contextual.unscored_claims", "{}"),
    ("contradições conceito/natureza", "diag:contextual.region_kind_contradictions", "{}"),
    ("regiões reconciliadas", "diag:contextual.reconciled_regions", "{}"),
    ("hipóteses afirmadas concorrentes", "diag:contextual.competing_assertions", "{}"),
    ("sem suporte independente", "diag:contextual.regions_with_unsupported_primary", "{}"),
    ("identidade ambígua", "diag:contextual.regions_with_ambiguous_identity", "{}"),
    ("grupos de superfície", "diag:contextual.entity_groups", "{}"),
    ("regiões em grupo", "diag:contextual.regions_in_entity_groups", "{}"),
    ("grupos corroborados", "diag:contextual.supported_entity_groups", "{}"),
    ("relações geométricas", "diag:contextual.geometric_relations", "{}"),
    ("falhas de interpretação", "man:interpretation_failure_count", "{}"),
    ("falhas de evidência", "man:evidence_failure_count", "{}"),
    ("falhas de sinal", "man:signal_failure_count", "{}"),
    ("falhas de relação", "man:relation_failure_count", "{}"),
    ("audit errors", "man:audit_error_count", "{}"),
    ("audit warnings", "man:audit_warning_count", "{}"),
    ("latência (s)", "man:latency_s", "{:.0f}"),
    ("chamadas de modelo (total)", "man:model_calls.total", "{}"),
)


# Resolve um caminho pontuado dentro de um dict aninhado. Existe para que a
# tabela de eixos seja declarativa: um eixo é um caminho, e não um trecho de
# código por campo.
def dig(payload: Any, path: str) -> Any:
    """Retorna o valor em ``path`` (``a.b.c``), ou ``None`` se algum nível faltar."""
    current = payload
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


# Carrega o diagnóstico e o registro de manifest de um frame. Devolve
# ``(None, None)`` quando o run não contém aquele frame, para que a comparação
# reporte a ausência em vez de falhar.
def load_frame(run: Path, frame_id: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Retorna ``(diagnostics, registro do manifest)`` de um frame, ou ``(None, None)``."""
    diagnostics_path = run / "frames" / frame_id / "diagnostics.json"
    manifest_path = run / "manifest.json"
    if not diagnostics_path.is_file() or not manifest_path.is_file():
        return None, None
    manifest = json.loads(manifest_path.read_text())
    record = next((f for f in manifest["frames"] if f.get("frame_id") == frame_id), None)
    return json.loads(diagnostics_path.read_text()), record


# Formata um valor para a tabela, preservando a distinção entre "ausente" e
# "zero". Um campo que não existe naquele run vira ``n/a``, e nunca 0: os dois
# significam coisas diferentes ao comparar uma capacidade nova.
def render(value: Any, fmt: str) -> str:
    """Formata um valor de eixo, ou ``n/a`` quando ele não existe naquele run."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "sim" if value else "não"
    try:
        return fmt.format(value)
    except (TypeError, ValueError):
        return str(value)


# Monta a tabela de um frame. Chamada por ``compare`` uma vez por frame.
def compare_frame(
    baseline: Path, candidate: Path, frame_id: str
) -> list[str]:
    """Retorna as linhas markdown da comparação de um frame."""
    base_diag, base_man = load_frame(baseline, frame_id)
    cand_diag, cand_man = load_frame(candidate, frame_id)
    lines = [f"### {frame_id}", ""]
    if cand_diag is None:
        return [*lines, "_ausente no run candidato._", ""]

    base_sha = (base_man or {}).get("input", {}).get("sha256")
    cand_sha = (cand_man or {}).get("input", {}).get("sha256")
    same_input = "sim" if base_sha and base_sha == cand_sha else "NÃO"
    lines += [
        f"| eixo | {baseline.name} | {candidate.name} |",
        "| --- | ---: | ---: |",
        f"| mesma entrada (SHA-256) | — | {same_input} |",
    ]
    for label, path, fmt in _AXES:
        scope, _, field = path.partition(":")
        base_source = base_diag if scope == "diag" else base_man
        cand_source = cand_diag if scope == "diag" else cand_man
        base_value = None if base_source is None else dig(base_source, field)
        cand_value = None if cand_source is None else dig(cand_source, field)
        lines.append(f"| {label} | {render(base_value, fmt)} | {render(cand_value, fmt)} |")

    semantic = dig(cand_diag, "contextual.semantic_relations") or []
    if semantic:
        rendered = ", ".join(f"`{name}` × {count}" for name, count in semantic)
        lines += ["", f"Relações semânticas: {rendered}."]
    else:
        lines += ["", "Relações semânticas: **nenhuma**."]

    calls = (cand_man or {}).get("model_calls", {})
    stages = ("hypothesis_support_text", "region_refinement", "semantic_relations")
    if any(calls.get(stage) for stage in stages):
        rendered = ", ".join(f"{stage}={calls.get(stage, 0)}" for stage in stages)
        lines.append(f"Custo dos estágios novos: {rendered}.")

    for step in (cand_man or {}).get("refinement", []):
        reasons: dict[str, int] = {}
        for target in step["targets"]:
            for reason in target["reasons"]:
                reasons[reason] = reasons.get(reason, 0) + 1
        lines.append(
            f"Refinamento it{step['iteration']}: {len(step['targets'])} alvos, "
            f"{len(step['refined_region_ids'])} reinterpretadas, "
            f"evidência {list(step['previous_evidence'])} → {list(step['new_evidence'])}, "
            f"razões={reasons}."
        )
    lines.append("")
    return lines


# Ponto de entrada da comparação. Emite markdown para stdout, e opcionalmente
# para um arquivo versionável ao lado dos runs comparados.
def compare(baseline: Path, candidate: Path, frame_ids: tuple[str, ...]) -> str:
    """Retorna o relatório markdown comparando os dois runs."""
    base_manifest = json.loads((baseline / "manifest.json").read_text())
    cand_manifest = json.loads((candidate / "manifest.json").read_text())
    lines = [
        f"# Comparação: {baseline.name} → {candidate.name}",
        "",
        "| | baseline | candidato |",
        "| --- | --- | --- |",
        f"| revisão | `{base_manifest.get('git_revision')}` | `{cand_manifest.get('git_revision')}` |",
        f"| prompt_version | `{base_manifest['config']['multimodal_reasoning']['prompt_version']}` "
        f"| `{cand_manifest['config']['multimodal_reasoning']['prompt_version']}` |",
        f"| perfil | `{base_manifest.get('context_profile')}` | `{cand_manifest.get('context_profile')}` |",
        f"| pico de VRAM (GiB) | "
        f"{max(f['peak_vram_bytes'] for f in base_manifest['frames']) / 2**30:.2f} | "
        f"{max(f['peak_vram_bytes'] for f in cand_manifest['frames']) / 2**30:.2f} |",
        "",
        "> Comparação **estrutural**. Sem anotação humana revisada (#210/#211) nenhuma",
        "> afirmação de acurácia semântica é possível, e nenhuma é feita aqui.",
        "",
    ]
    for frame_id in frame_ids:
        lines += compare_frame(baseline, candidate, frame_id)
    return "\n".join(lines)


# Compõe a CLI. A seleção de frames é explícita para que uma comparação possa
# ser repetida exatamente.
def main(argv: list[str] | None = None) -> int:
    """Compara dois runs e escreve o relatório."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--frame-id", action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = compare(args.baseline, args.candidate, tuple(args.frame_id) or BINDING_FRAMES)
    print(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nRelatório: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - entrypoint de linha de comando
    raise SystemExit(main())
