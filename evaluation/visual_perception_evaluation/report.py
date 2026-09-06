"""Renderização de relatórios de avaliação reproduzíveis.

Issues: #198 (relatório de métrica), #200 (relatório de ablation).

O renderizador aplica, na saída, a regra que o resto da avaliação aplica no
cálculo: uma métrica sem suporte aparece como ``não medida`` e nunca entra
em uma comparação entre configurações. Isso satisfaz literalmente o critério
da #198 — um relatório não pode apresentar uma métrica sem suporte como
evidência de superioridade semântica.
"""

from __future__ import annotations

from collections.abc import Sequence

from .metrics import MetricValue
from .suite import EvaluationReport

#: Métricas em que um valor *menor* é melhor. Usada para orientar a
#: comparação entre configurações sem que cada chamador precise saber.
LOWER_IS_BETTER = frozenset(
    {
        "expected_calibration_error",
        "brier_score",
        "selective_risk",
        "unscored_rate",
        "unsupported_claim_rate",
        "sample_failure_rate",
        "mean_latency_s",
        "peak_vram_bytes",
        "mean_model_calls",
    }
)


# Formata uma métrica para exibição, distinguindo "não medida" de zero.
# Existe para que a distinção sobreviva à renderização, que é onde ela
# costuma se perder.
def format_metric(metric: MetricValue) -> str:
    """Formata uma métrica com o seu suporte, ou como ``não medida``."""
    if not metric.measured:
        return "não medida (sem suporte)"
    suffix = f" (n={metric.support}"
    suffix += f", {metric.detail})" if metric.detail else ")"
    return f"{metric.value:.4f}{suffix}"


# Compara duas configurações em uma métrica, recusando a comparação quando
# algum dos lados não foi medido. Existe como a única implementação dessa
# regra, para que o report de ablation não a reimplemente com um `or 0.0`.
def compare(name: str, baseline: MetricValue, candidate: MetricValue) -> str:
    """Descreve a diferença entre duas medições, ou por que ela não é comparável."""
    if not baseline.measured or not candidate.measured:
        return "não comparável (uma das configurações não mediu esta métrica)"
    delta = (candidate.value or 0.0) - (baseline.value or 0.0)
    better = delta < 0 if name in LOWER_IS_BETTER else delta > 0
    direction = "melhor" if better else ("igual" if delta == 0.0 else "pior")
    return f"{delta:+.4f} ({direction})"


# Renderiza o relatório de uma configuração em Markdown. Existe para que o
# resultado de uma execução seja legível por humanos sem uma ferramenta
# extra, mantendo os mesmos números do objeto estruturado.
def render_report(report: EvaluationReport) -> str:
    """Renderiza um :class:`EvaluationReport` em Markdown."""
    lines = [
        f"# Avaliação — {report.configuration}",
        "",
        "## Proveniência",
        "",
        f"- conjunto de referência: `{report.reference_id}`",
        f"- run: `{report.run_id}`",
        f"- split: `{report.split.value}`",
        f"- fingerprint de configuração: `{report.config_fingerprint}`",
        f"- revisão de código: `{report.code_revision}`",
        "",
        "## Cobertura",
        "",
        f"- amostras avaliadas: {report.scored_sample_count} de {report.sample_count}",
        f"- amostras com falha: {report.failed_sample_count}",
        f"- anotações ignoradas: {report.ignored_annotation_count}",
        "",
        "## Métricas",
        "",
        "| métrica | valor | intervalo (bootstrap 95%) |",
        "| --- | --- | --- |",
    ]
    for name, metric in sorted(report.metrics.items()):
        interval = report.intervals.get(name)
        rendered = "—" if interval is None else f"[{interval[0]:.4f}, {interval[1]:.4f}]"
        lines.append(f"| `{name}` | {format_metric(metric)} | {rendered} |")

    lines += ["", "## Calibração", "", "| modo | ECE | Brier |", "| --- | --- | --- |"]
    for mode, metrics in report.calibration.items():
        lines.append(
            f"| `{mode}` | {format_metric(metrics['expected_calibration_error'])} "
            f"| {format_metric(metrics['brier_score'])} |"
        )

    lines += ["", "## Abstenção", ""]
    lines += [f"- `{name}`: {format_metric(metric)}" for name, metric in sorted(report.abstention.items())]

    lines += ["", "## Custo", ""]
    lines += [f"- `{name}`: {format_metric(metric)}" for name, metric in sorted(report.cost.items())]

    if report.strata:
        lines += [
            "",
            "## Estratos",
            "",
            "| estrato | recall | IoU | boundary F1 | top-1 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for stratum, metrics in report.strata.items():
            lines.append(
                f"| `{stratum}` | {format_metric(metrics['region_recall'])} "
                f"| {format_metric(metrics['matched_mask_iou'])} "
                f"| {format_metric(metrics['boundary_f1'])} "
                f"| {format_metric(metrics['label_top1'])} |"
            )

    if report.unsupported_claims:
        lines += ["", "## Claims sem suporte (amostra auditável)", ""]
        for claim in report.unsupported_claims[:20]:
            lines.append(
                f"- `{claim.sample_id}` / `{claim.predicted_region_id}`: "
                f"{claim.value!r} — {claim.reason}"
            )
    return "\n".join(lines) + "\n"


# Renderiza a comparação entre uma configuração baseline e as candidatas.
# Existe para o relatório de ablation (#200), que precisa mostrar o que cada
# mecanismo mudou, incluindo quando não mudou nada mensurável.
def render_comparison(baseline: EvaluationReport, candidates: Sequence[EvaluationReport]) -> str:
    """Renderiza uma tabela comparativa entre o baseline e as configurações candidatas."""
    names = sorted(
        {name for report in (baseline, *candidates) for name in report.metrics}
        | {f"cost:{name}" for report in (baseline, *candidates) for name in report.cost}
    )
    header = ["| métrica | " + " | ".join(
        [f"`{baseline.configuration}` (baseline)"] + [f"`{report.configuration}`" for report in candidates]
    ) + " |"]
    header.append("| --- | " + " | ".join(["---"] * (len(candidates) + 1)) + " |")

    rows: list[str] = []
    for name in names:
        base_metric = _lookup(baseline, name)
        cells = [format_metric(base_metric)]
        for report in candidates:
            candidate = _lookup(report, name)
            cells.append(f"{format_metric(candidate)}<br>{compare(_bare(name), base_metric, candidate)}")
        rows.append(f"| `{name}` | " + " | ".join(cells) + " |")

    return "\n".join(["## Comparação de ablation", "", *header, *rows]) + "\n"


# Busca uma métrica no relatório, aceitando o prefixo ``cost:`` para as
# métricas de custo. Helper de render_comparison.
def _lookup(report: EvaluationReport, name: str) -> MetricValue:
    """Retorna a métrica pedida, ou uma métrica sem suporte quando ausente."""
    if name.startswith("cost:"):
        return report.cost.get(_bare(name), MetricValue(_bare(name), None, 0))
    return report.metrics.get(name, MetricValue(name, None, 0))


# Remove o prefixo de namespace do nome de uma métrica.
def _bare(name: str) -> str:
    """Retorna o nome da métrica sem o prefixo ``cost:``."""
    return name.split(":", 1)[1] if name.startswith("cost:") else name
