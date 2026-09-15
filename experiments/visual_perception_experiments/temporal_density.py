"""Análise de densidade temporal de um run de visual-perception sobre uma janela de bag.

O experimento que este módulo serve pergunta uma coisa só: **uma amostragem
temporal regular e mais densa recupera evidência contextual que keyframes
esparsos estavam perdendo?** Para responder isso é preciso ler um run pelo eixo
do tempo, e não frame a frame isoladamente — que é exatamente o que
``manifest.json`` não permite, porque ele não registra timestamp nenhum: a única
identidade que sobrevive da extração até o run é o ``sequence_index`` embutido no
``frame_id``. O tempo vive no artifact de janela produzido por ``bag-window``.

Este módulo faz a junção das duas fontes e nada além disso. Ele é estritamente
read-only: não executa modelo, não reescreve run, não corrige observação. Em
particular **não implementa tracking temporal**. O que ele chama de
"persistência" é co-ocorrência de um conceito em frames vizinhos, não identidade
de objeto: ``wooden pallet`` em dez frames não é afirmação de dez pallets nem de
um só, e o relatório diz isso onde o número aparece.

Uso, a partir da raiz do repositório:

    python -m visual_perception_experiments.temporal_density \\
      --window artifacts/<trecho>-window.json \\
      --run modules/visual-perception/benchmarks/results/samples/<run> \\
      --baseline-window artifacts/<trecho-anterior>-window.json \\
      --baseline-run modules/visual-perception/benchmarks/results/samples/<run-anterior>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import RESULTS_ROOT

#: Papel da claim escrita pela reconciliação intra-frame. Quando existe, é ela
#: que representa o que a região afirma depois de as hipóteses concorrentes
#: terem sido resolvidas — e não a primeira ``primary``, que é apenas uma das
#: entradas dessa resolução.
RECONCILED_ROLE = "reconciled"

#: Papel das claims produzidas pelo raciocínio de região. Uma região fundida a
#: partir de várias proposals carrega mais de uma, uma por hipótese.
PRIMARY_ROLE = "primary"

#: Estágio que registra o conceito canônico usado pela política de publicação.
#: É lido apenas para reportar divergência em relação ao conceito afirmado, e
#: nunca substitui a claim reconciliada.
PUBLICATION_STAGE = "contextual_publication"

#: Vocabulário usado **exclusivamente** para escolher quais frames o relatório
#: destaca para inspeção humana. Não é política, não filtra nada e não entra em
#: nenhuma contagem: existe porque "o frame com mais dano" precisa de uma
#: definição explícita para ser reproduzível.
DAMAGE_HINTS = (
    "broken",
    "crack",
    "damage",
    "debris",
    "rubble",
    "stain",
    "water",
    "burn",
    "graffiti",
    "hole",
    "rust",
    "leak",
    "mold",
    "collapse",
    "peeling",
)

#: Mesmo papel de ``DAMAGE_HINTS``, para a pergunta sobre a porta no fim do
#: corredor.
DOOR_HINTS = ("door",)


# Descreve o que uma região publicada afirma, junto do que foi preciso descartar
# para chegar a essa afirmação. Existe porque uma região fundida carrega várias
# claims ``primary`` concorrentes, e reportar só a primeira esconderia
# exatamente a ambiguidade que este experimento precisa medir.
@dataclass(frozen=True)
class PublishedConcept:
    """Conceito afirmado por uma região publicada em um frame.

    Argumentos:
        region_id: identidade da região dentro da observação.
        concept: conceito afirmado, já resolvido pela reconciliação quando houve.
        category: categoria declarada junto do conceito.
        region_kind: ``thing`` ou ``stuff``, conforme a observação.
        confidence: confiança semântica declarada, ou ``None`` quando ausente.
        box: caixa da região em pixels, na ordem ``(x_min, y_min, x_max, y_max)``.
        area_px: área da caixa em pixels, usada para separar região pequena de
            região que ocupa quase o frame inteiro.
        canonical_concept: conceito canônico registrado pela publicação contextual.
        entity_id: grupo de entidade que absorveu a região, quando existe.
        competing: demais conceitos ``primary`` da mesma região, quando divergem.
    """

    region_id: str
    concept: str
    category: str | None
    region_kind: str | None
    confidence: float | None
    box: tuple[float, float, float, float]
    area_px: float
    canonical_concept: str | None
    entity_id: str | None
    competing: tuple[str, ...]

    # Expõe a ambiguidade como pergunta direta, para que o relatório não precise
    # reimplementar o critério em cada lugar que a menciona.
    @property
    def ambiguous(self) -> bool:
        """Indica se a região carregava conceitos ``primary`` divergentes."""
        return bool(self.competing)


# Reúne o que um único frame observou, já posicionado no tempo. Existe para que
# toda a análise posterior opere sobre uma sequência ordenada e não precise
# reabrir arquivos nem reconsultar a janela.
@dataclass(frozen=True)
class FrameObservation:
    """Observação de um frame, posicionada na janela temporal.

    Argumentos:
        frame_id: identidade do frame nos artifacts do run.
        sequence_index: posição original no stream RGB do bag.
        relative_time_s: instante do frame relativo ao início da janela.
        proposal_count: proposals descobertas antes da fusão.
        region_count: regiões canônicas da observação, publicadas ou não.
        published_region_count: regiões que a política contextual publicou.
        structural_context_count: regiões retidas como contexto estrutural.
        published: conceitos publicados, na ordem em que a observação os lista.
        structural_concepts: conceitos estruturais retidos e sua contagem.
        entity_group_count: hipóteses de entidade formadas no frame.
        semantic_confidence_degenerate: se o run marcou a confiança como degenerada.
        latency_s: tempo de processamento do frame.
    """

    frame_id: str
    sequence_index: int
    relative_time_s: float
    proposal_count: int
    region_count: int
    published_region_count: int
    structural_context_count: int
    published: tuple[PublishedConcept, ...]
    structural_concepts: tuple[tuple[str, int], ...]
    entity_group_count: int
    semantic_confidence_degenerate: bool
    latency_s: float


# Resume a trajetória de um conceito ao longo da sequência. Existe porque a
# pergunta do experimento é temporal: um conceito que aparece em um frame
# isolado e um que aparece em todos os frames de uma passagem são evidências
# muito diferentes, e a contagem total não distingue os dois.
@dataclass(frozen=True)
class ConceptPersistence:
    """Trajetória temporal de um conceito publicado.

    Argumentos:
        concept: conceito publicado.
        categories: categorias sob as quais ele foi publicado.
        first_seen_s: instante da primeira publicação.
        last_seen_s: instante da última publicação.
        first_index: posição do primeiro frame que o publicou.
        last_index: posição do último frame que o publicou.
        frame_count: quantidade de frames que o publicaram.
        span_frames: frames entre a primeira e a última publicação, inclusive.
        persistence: ``frame_count / span_frames``; 1.0 é presença contínua.
        region_count: total de regiões publicadas com esse conceito.
        max_area_px: maior caixa publicada com esse conceito.
        frame_ids: frames que o publicaram, em ordem cronológica.
    """

    concept: str
    categories: tuple[str, ...]
    first_seen_s: float
    last_seen_s: float
    first_index: int
    last_index: int
    frame_count: int
    span_frames: int
    persistence: float
    region_count: int
    max_area_px: float
    frame_ids: tuple[str, ...]


# Agrega uma sequência inteira nos eixos que a comparação entre políticas de
# amostragem precisa. Existe para que "mais denso é melhor?" seja respondido por
# uma tabela e não por impressão.
@dataclass(frozen=True)
class TimelineTotals:
    """Totais de uma sequência de frames.

    Argumentos:
        frame_count: frames processados.
        span_s: intervalo coberto, do primeiro ao último frame.
        proposal_total: soma das proposals.
        region_total: soma das regiões canônicas.
        published_total: soma das regiões publicadas.
        structural_total: soma do contexto estrutural.
        empty_frames: frames sem nenhuma evidência contextual.
        longest_empty_run: maior sequência consecutiva de frames sem evidência.
        distinct_concepts: conceitos publicados distintos.
        ambiguous_regions: regiões publicadas com conceitos ``primary`` divergentes.
        concept_counts: contagem de regiões por conceito publicado.
    """

    frame_count: int
    span_s: float
    proposal_total: int
    region_total: int
    published_total: int
    structural_total: int
    empty_frames: int
    longest_empty_run: int
    distinct_concepts: int
    ambiguous_regions: int
    concept_counts: tuple[tuple[str, int], ...]


# Lê a claim que representa o que uma região publicada afirma. A regra é
# explícita porque a estrutura permite mais de uma resposta: a reconciliação
# intra-frame, quando roda, é a autoridade; sem ela, vale a única ``primary``.
# Quando há várias ``primary`` divergentes e nenhuma reconciliação, a primeira é
# devolvida e as demais ficam registradas em ``competing`` — escolher em
# silêncio esconderia a ambiguidade que este experimento precisa observar.
def published_concept_of(
    region: dict[str, Any], entity_by_region: dict[str, str] | None = None
) -> PublishedConcept:
    """Resolve o conceito publicado por uma região da observação.

    Argumentos:
        region: região publicada, como serializada em ``observation.json``.
        entity_by_region: mapa de região para grupo de entidade, quando disponível.
    Retorna:
        o conceito publicado, com as hipóteses concorrentes preservadas.
    Levanta:
        ValueError: se a região não declarar nenhuma claim utilizável.
    """
    claims = region.get("claims") or []
    primaries = [claim for claim in claims if claim.get("role") == PRIMARY_ROLE]
    reconciled = next((claim for claim in claims if claim.get("role") == RECONCILED_ROLE), None)
    chosen = reconciled or (primaries[0] if primaries else None)
    if chosen is None:
        raise ValueError(f"region {region.get('region_id')} declares no primary or reconciled claim.")

    distinct = tuple(dict.fromkeys(str(claim["value"]) for claim in primaries))
    competing = tuple(value for value in distinct if value != str(chosen["value"]))
    canonical = next(
        (
            str(claim["value"])
            for claim in claims
            if (claim.get("provenance") or {}).get("stage") == PUBLICATION_STAGE
        ),
        None,
    )
    box = region.get("box") or {}
    corners = (
        float(box.get("x_min", 0.0)),
        float(box.get("y_min", 0.0)),
        float(box.get("x_max", 0.0)),
        float(box.get("y_max", 0.0)),
    )
    confidence = chosen.get("confidence") or {}
    region_id = str(region.get("region_id", ""))
    return PublishedConcept(
        region_id=region_id,
        concept=str(chosen["value"]),
        category=None if chosen.get("category") is None else str(chosen["category"]),
        region_kind=None if chosen.get("region_kind") is None else str(chosen["region_kind"]),
        confidence=None if confidence.get("value") is None else float(confidence["value"]),
        box=corners,
        area_px=max(corners[2] - corners[0], 0.0) * max(corners[3] - corners[1], 0.0),
        canonical_concept=canonical,
        entity_id=(entity_by_region or {}).get(region_id),
        competing=competing,
    )


# Inverte as hipóteses de entidade em um mapa de região para grupo. Existe
# porque a pergunta "as duas regiões de broken tile foram reconhecidas como a
# mesma entidade?" é feita por região, e a observação registra por grupo.
def entity_by_region_id(observation: dict[str, Any]) -> dict[str, str]:
    """Mapeia cada região ao grupo de entidade que a absorveu.

    Argumentos:
        observation: observação serializada de um frame.
    Retorna:
        mapa de ``region_id`` para ``entity_id``.
    """
    mapping: dict[str, str] = {}
    for hypothesis in observation.get("entity_hypotheses") or []:
        for region_id in hypothesis.get("member_region_ids") or []:
            mapping[str(region_id)] = str(hypothesis["entity_id"])
    return mapping


# Conta os conceitos retidos como contexto estrutural. Fica separado porque o
# contexto estrutural usa a mesma estrutura de claims das regiões publicadas, e
# duplicar a leitura divergiria com o tempo.
def structural_concept_counts(observation: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    """Conta os conceitos retidos como contexto estrutural em um frame.

    Argumentos:
        observation: observação serializada de um frame.
    Retorna:
        pares ``(conceito, contagem)`` em ordem decrescente de contagem.
    """
    counter: Counter[str] = Counter()
    for region in observation.get("structural_context") or []:
        try:
            counter[published_concept_of(region).concept] += 1
        except ValueError:
            continue
    return tuple(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


# Junta a janela temporal com os artifacts de um run e devolve a sequência
# ordenada no tempo. É o ponto de entrada da análise: sem esta junção, o run não
# tem eixo temporal, porque nenhum artifact do run registra timestamp.
def build_timeline(window: Path, run: Path) -> tuple[FrameObservation, ...]:
    """Constrói a sequência temporal de um run sobre uma janela resolvida.

    Argumentos:
        window: artifact JSON produzido por ``mapping-runtime bag-window``.
        run: diretório do run, contendo ``manifest.json`` e ``frames/``.
    Retorna:
        observações em ordem cronológica crescente.
    Levanta:
        ValueError: se nenhum keyframe da janela tiver artifacts no run.
    """
    payload = json.loads(window.read_text(encoding="utf-8"))
    prefix = str(payload.get("frame_id_prefix", "corridor-02"))
    start_ns = int(payload["start_header_ns"])
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    reports = {str(report["frame_id"]): report for report in manifest.get("frames", [])}

    observations: list[FrameObservation] = []
    for entry in payload["keyframes"]:
        index = int(entry["sequence_index"])
        frame_id = f"{prefix}-{index:05d}"
        report = reports.get(frame_id)
        frame_dir = run / "frames" / frame_id
        if report is None or report.get("failed") or not (frame_dir / "observation.json").is_file():
            continue
        observation = json.loads((frame_dir / "observation.json").read_text(encoding="utf-8"))
        diagnostics = json.loads((frame_dir / "diagnostics.json").read_text(encoding="utf-8"))
        entities = entity_by_region_id(observation)
        published = tuple(
            published_concept_of(region, entities) for region in observation.get("regions") or []
        )
        observations.append(
            FrameObservation(
                frame_id=frame_id,
                sequence_index=index,
                relative_time_s=(int(entry["header_timestamp_ns"]) - start_ns) / 1_000_000_000,
                proposal_count=int(report["proposal_count"]),
                region_count=int(report["canonical_region_count"]),
                published_region_count=int(report["published_region_count"]),
                structural_context_count=int(report["structural_context_count"]),
                published=published,
                structural_concepts=structural_concept_counts(observation),
                entity_group_count=len(observation.get("entity_hypotheses") or []),
                semantic_confidence_degenerate=bool(
                    (diagnostics.get("semantic_confidence") or {}).get("degenerate", False)
                ),
                latency_s=float(report.get("latency_s", 0.0)),
            )
        )
    if not observations:
        raise ValueError(f"no keyframe of {window} has perception artifacts under {run / 'frames'}.")
    return tuple(sorted(observations, key=lambda item: item.relative_time_s))


# Recorta uma sequência a um intervalo temporal. Existe porque comparar 30 s
# amostrados a 0,5 FPS com 15 s amostrados a 1 FPS mede duas coisas ao mesmo
# tempo — densidade e extensão do trecho. Recortar o baseline ao trecho do
# candidato isola a densidade, que é a variável do experimento.
def restrict_to_span(
    timeline: Sequence[FrameObservation], start_s: float, end_s: float
) -> tuple[FrameObservation, ...]:
    """Recorta a sequência ao intervalo temporal informado, inclusive nas pontas.

    Argumentos:
        timeline: sequência ordenada de observações.
        start_s: início do intervalo, em segundos relativos à janela.
        end_s: fim do intervalo, em segundos relativos à janela.
    Retorna:
        as observações dentro do intervalo, preservando a ordem.
    Levanta:
        ValueError: se o intervalo for invertido.
    """
    if end_s < start_s:
        raise ValueError("end_s must not precede start_s.")
    # A tolerância absorve a diferença entre o instante pedido e o header do
    # frame realmente mais próximo, que chega a algumas dezenas de ms.
    tolerance = 0.05
    return tuple(
        frame for frame in timeline if start_s - tolerance <= frame.relative_time_s <= end_s + tolerance
    )


# Mede a trajetória temporal de cada conceito publicado. Reporta co-ocorrência,
# não identidade: agrupar por conceito é o mais forte que se pode afirmar sem
# tracking temporal, que é deliberadamente uma etapa posterior.
def concept_persistence(timeline: Sequence[FrameObservation]) -> tuple[ConceptPersistence, ...]:
    """Resume a persistência temporal de cada conceito publicado.

    Argumentos:
        timeline: sequência ordenada de observações.
    Retorna:
        persistências em ordem decrescente de frames que publicaram o conceito.
    """
    positions: dict[str, list[int]] = {}
    regions: Counter[str] = Counter()
    categories: dict[str, list[str]] = {}
    areas: dict[str, float] = {}
    for position, frame in enumerate(timeline):
        for concept in frame.published:
            positions.setdefault(concept.concept, [])
            if not positions[concept.concept] or positions[concept.concept][-1] != position:
                positions[concept.concept].append(position)
            regions[concept.concept] += 1
            areas[concept.concept] = max(areas.get(concept.concept, 0.0), concept.area_px)
            if concept.category is not None:
                categories.setdefault(concept.concept, [])
                if concept.category not in categories[concept.concept]:
                    categories[concept.concept].append(concept.category)

    summaries: list[ConceptPersistence] = []
    for concept, seen in positions.items():
        span = seen[-1] - seen[0] + 1
        summaries.append(
            ConceptPersistence(
                concept=concept,
                categories=tuple(categories.get(concept, ())),
                first_seen_s=timeline[seen[0]].relative_time_s,
                last_seen_s=timeline[seen[-1]].relative_time_s,
                first_index=seen[0],
                last_index=seen[-1],
                frame_count=len(seen),
                span_frames=span,
                persistence=len(seen) / span,
                region_count=regions[concept],
                max_area_px=areas.get(concept, 0.0),
                frame_ids=tuple(timeline[position].frame_id for position in seen),
            )
        )
    return tuple(sorted(summaries, key=lambda item: (-item.frame_count, -item.region_count, item.concept)))


# Calcula os totais de uma sequência, incluindo a maior sequência consecutiva
# sem evidência contextual — que é a métrica que descreve um trecho de mapa
# ficando cego, e que a soma de frames vazios sozinha não revela.
def aggregate_totals(timeline: Sequence[FrameObservation]) -> TimelineTotals:
    """Agrega uma sequência nos eixos comparados entre políticas de amostragem.

    Argumentos:
        timeline: sequência ordenada de observações.
    Retorna:
        os totais da sequência.
    Levanta:
        ValueError: se a sequência estiver vazia.
    """
    if not timeline:
        raise ValueError("cannot aggregate an empty timeline.")
    counts: Counter[str] = Counter()
    ambiguous = 0
    longest = current = 0
    for frame in timeline:
        for concept in frame.published:
            counts[concept.concept] += 1
            ambiguous += 1 if concept.ambiguous else 0
        current = current + 1 if frame.published_region_count == 0 else 0
        longest = max(longest, current)
    return TimelineTotals(
        frame_count=len(timeline),
        span_s=timeline[-1].relative_time_s - timeline[0].relative_time_s,
        proposal_total=sum(frame.proposal_count for frame in timeline),
        region_total=sum(frame.region_count for frame in timeline),
        published_total=sum(frame.published_region_count for frame in timeline),
        structural_total=sum(frame.structural_context_count for frame in timeline),
        empty_frames=sum(1 for frame in timeline if frame.published_region_count == 0),
        longest_empty_run=longest,
        distinct_concepts=len(counts),
        ambiguous_regions=ambiguous,
        concept_counts=tuple(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    )


# Compara os frames que os dois runs têm em comum. Existe porque este é o único
# controle disponível de não-determinismo: os mesmos pixels, a mesma
# configuração, dois runs. Sem ele não há como saber quanto de uma diferença
# entre políticas de amostragem é apenas variação do pipeline.
def shared_frame_drift(
    baseline: Sequence[FrameObservation], candidate: Sequence[FrameObservation]
) -> tuple[tuple[str, int, int, tuple[str, ...], tuple[str, ...]], ...]:
    """Compara frames presentes nos dois runs, frame a frame.

    Argumentos:
        baseline: sequência do run de referência.
        candidate: sequência do run comparado.
    Retorna:
        tuplas ``(frame_id, publicadas no baseline, publicadas no candidato,
        conceitos só no baseline, conceitos só no candidato)``.
    """
    by_id = {frame.frame_id: frame for frame in candidate}
    rows = []
    for frame in baseline:
        other = by_id.get(frame.frame_id)
        if other is None:
            continue
        left = {concept.concept for concept in frame.published}
        right = {concept.concept for concept in other.published}
        rows.append(
            (
                frame.frame_id,
                frame.published_region_count,
                other.published_region_count,
                tuple(sorted(left - right)),
                tuple(sorted(right - left)),
            )
        )
    return tuple(rows)


# Seleciona os frames que merecem inspeção visual, por critérios declarados em
# vez de escolha manual. Existe para que o relatório aponte para artifacts
# concretos sem que alguém precise abrir os trinta overlays.
def interesting_frames(timeline: Sequence[FrameObservation]) -> tuple[tuple[str, FrameObservation], ...]:
    """Escolhe os frames de interesse por critérios reproduzíveis.

    Argumentos:
        timeline: sequência ordenada de observações.
    Retorna:
        pares ``(motivo, frame)``, sem repetir motivo.
    Levanta:
        ValueError: se a sequência estiver vazia.
    """
    if not timeline:
        raise ValueError("cannot select frames from an empty timeline.")

    def matching(hints: Iterable[str]) -> list[FrameObservation]:
        lowered = tuple(hints)
        return [
            frame
            for frame in timeline
            if any(hint in concept.concept.lower() for concept in frame.published for hint in lowered)
        ]

    selected: list[tuple[str, FrameObservation]] = [
        ("primeiro frame", timeline[0]),
        ("último frame", timeline[-1]),
        ("mais evidências publicadas", max(timeline, key=lambda frame: frame.published_region_count)),
    ]
    largest = max(
        timeline,
        key=lambda frame: max((concept.area_px for concept in frame.published), default=0.0),
    )
    if largest.published:
        selected.append(("maior região publicada (candidato a falso positivo)", largest))
    doors = matching(DOOR_HINTS)
    if doors:
        selected.append(
            ("door na maior área", max(doors, key=lambda frame: max(c.area_px for c in frame.published)))
        )
    damage = matching(DAMAGE_HINTS)
    if damage:
        selected.append(
            (
                "mais dano/detrito",
                max(
                    damage,
                    key=lambda frame: sum(
                        1
                        for concept in frame.published
                        if any(hint in concept.concept.lower() for hint in DAMAGE_HINTS)
                    ),
                ),
            )
        )
    return tuple(selected)


# Formata a lista de conceitos de um frame para uma célula de tabela.
def _concepts_cell(frame: FrameObservation) -> str:
    """Descreve os conceitos publicados de um frame em uma linha de tabela."""
    if not frame.published:
        return "—"
    parts = []
    for concept in frame.published:
        marker = " ⚠" if concept.ambiguous else ""
        parts.append(f"{concept.concept} ({int(concept.area_px)} px){marker}")
    return "; ".join(parts)


# Renderiza a seção de totais de uma ou mais sequências lado a lado.
def _totals_table(columns: Sequence[tuple[str, TimelineTotals]]) -> str:
    """Monta a tabela de totais com uma coluna por braço comparado."""
    header = "| eixo | " + " | ".join(label for label, _ in columns) + " |"
    divider = "|---" * (len(columns) + 1) + "|"
    rows = [
        ("frames", lambda t: str(t.frame_count)),
        ("trecho coberto (s)", lambda t: f"{t.span_s:.1f}"),
        ("proposals", lambda t: str(t.proposal_total)),
        ("regiões observadas", lambda t: str(t.region_total)),
        ("regiões publicadas", lambda t: str(t.published_total)),
        ("contexto estrutural", lambda t: str(t.structural_total)),
        ("publicadas por frame", lambda t: f"{t.published_total / t.frame_count:.2f}"),
        ("conceitos distintos", lambda t: str(t.distinct_concepts)),
        ("frames sem evidência contextual", lambda t: f"{t.empty_frames}/{t.frame_count}"),
        ("maior sequência cega", lambda t: str(t.longest_empty_run)),
        ("regiões com primárias divergentes", lambda t: str(t.ambiguous_regions)),
    ]
    lines = [header, divider]
    for label, render in rows:
        lines.append(f"| {label} | " + " | ".join(render(totals) for _, totals in columns) + " |")
    return "\n".join(lines)


# Monta o relatório de dados do experimento. É deliberadamente descritivo: ele
# apresenta o que os runs registraram e não conclui nada sobre correção
# semântica, que exigiria anotação humana que este conjunto não tem.
def render_report(
    candidate: Sequence[FrameObservation],
    *,
    candidate_label: str,
    candidate_run: Path,
    baseline: Sequence[FrameObservation] | None = None,
    baseline_label: str = "baseline",
    baseline_run: Path | None = None,
) -> str:
    """Renderiza o relatório de densidade temporal em markdown.

    Argumentos:
        candidate: sequência do run comparado.
        candidate_label: rótulo do braço candidato nas tabelas.
        candidate_run: diretório do run candidato, citado na proveniência.
        baseline: sequência do run de referência, quando houver.
        baseline_label: rótulo do braço de referência nas tabelas.
        baseline_run: diretório do run de referência, citado na proveniência.
    Retorna:
        o relatório em markdown.
    """
    totals = aggregate_totals(candidate)
    sections: list[str] = [
        "# Densidade temporal — relatório de dados",
        "",
        "Gerado por `visual_perception_experiments.temporal_density`. Estritamente descritivo: ",
        "apresenta o que os runs registraram sobre si mesmos, sem afirmar correção semântica — ",
        "o conjunto não tem anotação humana revisada, e usar a saída do próprio pipeline como ",
        "ground truth produziria uma acurácia sem significado.",
        "",
        "## Proveniência",
        "",
        f"- **{candidate_label}**: `{candidate_run}`",
    ]
    if baseline is not None and baseline_run is not None:
        sections.append(f"- **{baseline_label}**: `{baseline_run}`")
    sections += ["", "## Por frame", "", f"### {candidate_label}", ""]
    sections.append("| # | t (s) | frame | proposals | regiões | publicadas | estrutural | conceitos publicados |")
    sections.append("|---|---|---|---|---|---|---|---|")
    for position, frame in enumerate(candidate):
        sections.append(
            f"| {position:02d} | {frame.relative_time_s:6.2f} | `{frame.frame_id}` | "
            f"{frame.proposal_count} | {frame.region_count} | {frame.published_region_count} | "
            f"{frame.structural_context_count} | {_concepts_cell(frame)} |"
        )

    sections += ["", "## Totais", ""]
    columns: list[tuple[str, TimelineTotals]] = []
    if baseline is not None:
        columns.append((f"{baseline_label} (completo)", aggregate_totals(baseline)))
        half = restrict_to_span(baseline, candidate[0].relative_time_s, candidate[-1].relative_time_s)
        if half:
            columns.append((f"{baseline_label} (mesmo trecho)", aggregate_totals(half)))
    columns.append((candidate_label, totals))
    sections.append(_totals_table(columns))

    sections += ["", "## Conceitos publicados", "", "| conceito | regiões |", "|---|---|"]
    for concept, count in totals.concept_counts:
        sections.append(f"| {concept} | {count} |")

    sections += [
        "",
        "## Persistência temporal",
        "",
        "`persistência` é `frames / span`: 1.00 significa presença em todos os frames entre a ",
        "primeira e a última publicação. **Isto é co-ocorrência de conceito, não identidade de ",
        "objeto** — nenhum tracking temporal foi executado, então `n` frames não são `n` objetos ",
        "nem garantidamente o mesmo.",
        "",
        "| conceito | 1º visto (s) | último (s) | frames | span | persistência | regiões | maior área (px) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for summary in concept_persistence(candidate):
        sections.append(
            f"| {summary.concept} | {summary.first_seen_s:.2f} | {summary.last_seen_s:.2f} | "
            f"{summary.frame_count} | {summary.span_frames} | {summary.persistence:.2f} | "
            f"{summary.region_count} | {int(summary.max_area_px)} |"
        )

    if baseline is not None:
        drift = shared_frame_drift(baseline, candidate)
        sections += [
            "",
            "## Controle de não-determinismo",
            "",
            "Frames presentes nos dois runs — mesmos pixels, mesma configuração. A divergência ",
            "aqui é ruído do pipeline, e calibra quanto de qualquer diferença entre os braços é real.",
            "",
            "| frame | publicadas no baseline | publicadas no candidato | só no baseline | só no candidato |",
            "|---|---|---|---|---|",
        ]
        for frame_id, left, right, only_left, only_right in drift:
            sections.append(
                f"| `{frame_id}` | {left} | {right} | "
                f"{', '.join(only_left) or '—'} | {', '.join(only_right) or '—'} |"
            )

    sections += ["", "## Frames de interesse", "", "| motivo | frame | publicadas |", "|---|---|---|"]
    for reason, frame in interesting_frames(candidate):
        sections.append(f"| {reason} | `{frame.frame_id}` | {frame.published_region_count} |")

    degenerate = sum(1 for frame in candidate if frame.semantic_confidence_degenerate)
    sections += [
        "",
        "## Notas estruturais",
        "",
        f"- confiança semântica degenerada em {degenerate}/{len(candidate)} frames;",
        f"- {totals.ambiguous_regions} regiões publicadas carregam claims `primary` divergentes ",
        "  resolvidas pela reconciliação intra-frame (marcadas com ⚠ na tabela por frame);",
        f"- {sum(frame.entity_group_count for frame in candidate)} hipóteses de entidade formadas ao ",
        "  longo da sequência.",
        "",
    ]
    return "\n".join(sections)


# Constrói o parser em um helper testável, separado da execução.
def _argument_parser() -> argparse.ArgumentParser:
    """Retorna o parser da análise de densidade temporal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window", type=Path, required=True, help="window JSON do run analisado")
    parser.add_argument("--run", type=Path, required=True, help="diretório do run analisado")
    parser.add_argument("--label", default="candidato", help="rótulo do braço analisado")
    parser.add_argument("--baseline-window", type=Path, default=None)
    parser.add_argument("--baseline-run", type=Path, default=None)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="destino do relatório; o default fica sob experiments/results/temporal-density/",
    )
    return parser


# Ponto de entrada de CLI: resolve as duas sequências, renderiza o relatório e o
# persiste ao lado dos demais resultados de experimento.
def main(argv: list[str] | None = None) -> None:
    """Executa a análise de densidade temporal e grava o relatório."""
    arguments = _argument_parser().parse_args(argv)
    candidate = build_timeline(arguments.window, arguments.run)
    baseline = None
    if arguments.baseline_window is not None and arguments.baseline_run is not None:
        baseline = build_timeline(arguments.baseline_window, arguments.baseline_run)
    report = render_report(
        candidate,
        candidate_label=arguments.label,
        candidate_run=arguments.run,
        baseline=baseline,
        baseline_label=arguments.baseline_label,
        baseline_run=arguments.baseline_run,
    )
    destination = arguments.output or (
        RESULTS_ROOT / "temporal-density" / arguments.window.stem.replace("-window", "") / "report.md"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    print(destination)


if __name__ == "__main__":  # pragma: no cover - ponto de entrada de CLI
    main()
