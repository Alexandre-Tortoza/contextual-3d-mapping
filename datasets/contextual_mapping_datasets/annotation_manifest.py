"""Manifest versionado do conjunto de referência anotado de percepção visual.

Issue: #197.

Este manifest liga *amostras imutáveis* (um frame extraído, identificado pelo
digest do arquivo) às anotações que servem de referência para avaliação, e à
proveniência de quem anotou. Ele é de posse de `datasets/` porque descreve
dados, não algoritmo: o módulo de percepção consome as anotações, mas não
decide o que é uma anotação válida.

Três decisões estruturam o schema:

- **vocabulário aberto**: uma região carrega uma tupla de labels candidatos,
  não uma classe de um conjunto fechado. Ambiguidade é representada por
  ``certainty`` (``ambiguous``) e desconhecimento por ``unknown`` com labels
  vazios — nunca por um label inventado;
- **regiões ignoradas**: uma região pode ser marcada ``ignored``, o que a
  retira do cálculo de métrica sem transformá-la em fundo. É como
  degradação, oclusão e recorte de borda entram no conjunto sem penalizar
  injustamente uma predição correta;
- **estado de revisão**: uma amostra nasce ``pending_review``. A validação
  de release (``require_reviewed``) recusa um conjunto que ainda não passou
  por revisão humana, para que nenhum número seja publicado contra
  anotações não conferidas.

A validação recusa identidades duplicadas, vazamento de split (o mesmo
artifact ou o mesmo frame de origem aparecendo em mais de um split) e
referências de anotação penduradas.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from .paths import validate_dataset_name

#: Versão do schema deste manifest. Incrementar junto com uma migração
#: explícita, nunca em silêncio.
REFERENCE_SCHEMA_VERSION = "visual-reference/1"


# Enumera os splits do conjunto de referência. Existe para que a pertença de
# uma amostra seja um valor fechado e verificável, e não uma string livre
# como no manifest de dataset bruto (onde split é apenas informativo).
class Split(StrEnum):
    """A que subconjunto de avaliação uma amostra pertence."""

    DEVELOPMENT = "development"
    CALIBRATION = "calibration"
    TEST = "test"


# Distingue anotação conferida de anotação ainda não revisada. Existe para
# que um conjunto recém-gerado por ferramenta nunca seja confundido com um
# conjunto revisado por humano.
class ReviewState(StrEnum):
    """O estado de revisão humana de uma amostra anotada."""

    PENDING_REVIEW = "pending_review"
    REVIEWED = "reviewed"
    REJECTED = "rejected"


# Distingue uma anotação confiante de uma ambígua e de uma desconhecida.
# Existe porque a #197 exige que ambiguidade e desconhecimento sejam
# expressáveis sem forçar uma taxonomia fechada.
class AnnotationCertainty(StrEnum):
    """Quanta certeza o anotador teve sobre o conteúdo de uma anotação."""

    CERTAIN = "certain"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


# Referencia o arquivo imutável de uma amostra pelo seu digest. Existe para
# que uma amostra continue identificável mesmo se o caminho mudar, e para
# que a detecção de vazamento de split possa comparar conteúdo, não nome.
@dataclass(frozen=True)
class SampleArtifact:
    """O arquivo imutável de uma amostra, identificado pelo seu digest."""

    uri: str
    media_type: str
    digest: str

    # Exige uri, media_type e um digest SHA-256 hexadecimal, já que é o
    # digest que torna a amostra imutável e comparável entre splits.
    def __post_init__(self) -> None:
        """Rejeita artifacts sem uri, sem media_type ou com digest malformado."""
        for name in ("uri", "media_type", "digest"):
            if not getattr(self, name).strip():
                raise ValueError(f"SampleArtifact.{name} must not be empty.")
        if len(self.digest) != 64 or any(char not in "0123456789abcdef" for char in self.digest):
            raise ValueError("SampleArtifact.digest must be a lowercase hex SHA-256 digest.")


# Codifica a máscara de uma região anotada em RLE, mantendo o pacote livre de
# numpy. Existe para que o manifest possa ser lido por ferramentas leves
# (revisão, validação em CI) sem carregar a stack numérica.
@dataclass(frozen=True)
class MaskAnnotation:
    """Uma máscara booleana codificada em RLE, começando por ``False``."""

    width: int
    height: int
    runs: tuple[int, ...]

    # Valida resolução positiva e um RLE que cobre exatamente a imagem, para
    # que uma máscara truncada nunca vire uma região silenciosamente menor.
    def __post_init__(self) -> None:
        """Rejeita resoluções inválidas e RLE que não cobre a imagem inteira."""
        if self.width <= 0 or self.height <= 0:
            raise ValueError("MaskAnnotation dimensions must be positive.")
        if not self.runs or any(type(run) is not int or run < 0 for run in self.runs):
            raise ValueError("MaskAnnotation.runs must be non-negative integers.")
        if sum(self.runs) != self.width * self.height:
            raise ValueError(
                f"MaskAnnotation.runs must cover exactly {self.width * self.height} pixels, "
                f"got {sum(self.runs)}."
            )

    # Conta os pixels ocupados sem materializar a máscara. Usada pelos
    # estratos de tamanho de região, que só precisam da área.
    @property
    def area(self) -> int:
        """Número de pixels ocupados, derivado direto do RLE."""
        return sum(run for index, run in enumerate(self.runs) if index % 2 == 1)


# Representa uma região anotada com semântica de vocabulário aberto. Existe
# como a unidade de referência contra a qual as predições de região do módulo
# são comparadas.
@dataclass(frozen=True)
class RegionAnnotation:
    """Uma região anotada, com labels candidatos e atributos opcionais."""

    annotation_id: str
    mask: MaskAnnotation
    labels: tuple[str, ...] = field(default_factory=tuple)
    certainty: AnnotationCertainty = AnnotationCertainty.CERTAIN
    attributes: tuple[str, ...] = field(default_factory=tuple)
    condition: str | None = None
    material: str | None = None
    hazards: tuple[str, ...] = field(default_factory=tuple)
    visibility: str | None = None
    ignored: bool = False
    notes: str | None = None

    # Impõe a regra que mantém o vocabulário aberto sem abrir espaço para
    # label inventado: só uma anotação ``unknown`` (ou ignorada) pode não ter
    # nenhum label, e nenhuma anotação pode ter label vazio.
    def __post_init__(self) -> None:
        """Valida identidade e a coerência entre certeza e labels declarados."""
        if not self.annotation_id.strip():
            raise ValueError("RegionAnnotation.annotation_id must not be empty.")
        if any(not label.strip() for label in self.labels):
            raise ValueError(f"RegionAnnotation({self.annotation_id!r}) has an empty label.")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"RegionAnnotation({self.annotation_id!r}) repeats a label.")
        unlabelled_is_allowed = self.certainty is AnnotationCertainty.UNKNOWN or self.ignored
        if not self.labels and not unlabelled_is_allowed:
            raise ValueError(
                f"RegionAnnotation({self.annotation_id!r}) has no label: mark it 'unknown' or "
                "'ignored' instead of leaving the semantics implicit."
            )
        if self.certainty is AnnotationCertainty.UNKNOWN and self.labels:
            raise ValueError(
                f"RegionAnnotation({self.annotation_id!r}) is 'unknown' but declares labels."
            )


# Representa uma relação anotada entre duas regiões da mesma amostra.
# Existe para que as métricas de relação (#198) tenham uma referência, e para
# que referências penduradas sejam detectáveis na validação.
@dataclass(frozen=True)
class RelationAnnotation:
    """Uma relação anotada entre duas regiões da mesma amostra."""

    relation_id: str
    subject_annotation_id: str
    predicate: str
    object_annotation_id: str
    certainty: AnnotationCertainty = AnnotationCertainty.CERTAIN

    # Valida identidade, predicate e rejeita auto-relações, que nunca são
    # informação útil de referência.
    def __post_init__(self) -> None:
        """Rejeita ids vazios, predicate vazio e auto-relações."""
        for name in ("relation_id", "subject_annotation_id", "predicate", "object_annotation_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"RelationAnnotation.{name} must not be empty.")
        if self.subject_annotation_id == self.object_annotation_id:
            raise ValueError(
                f"RelationAnnotation({self.relation_id!r}) is a self-relation, which carries no "
                "reference information."
            )


# Registra quem anotou, com que ferramenta e sob qual política de resolução
# de divergência. Existe para que um número de avaliação possa ser rastreado
# até as condições em que sua referência foi produzida.
@dataclass(frozen=True)
class AnnotationProvenance:
    """Quem produziu uma anotação, com que método e sob qual política."""

    annotator: str
    method: str
    policy_version: str
    annotated_at: str | None = None
    resolution: str | None = None

    # Exige os três campos sem os quais a anotação não é rastreável.
    def __post_init__(self) -> None:
        """Rejeita proveniência sem anotador, método ou versão de política."""
        for name in ("annotator", "method", "policy_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"AnnotationProvenance.{name} must not be empty.")


# Agrupa tudo que descreve uma amostra do conjunto de referência: sua
# identidade, o artifact imutável, o split, as condições de captura, as
# anotações e a proveniência. É a unidade que as métricas (#198) consomem.
@dataclass(frozen=True)
class SampleAnnotation:
    """Uma amostra do conjunto de referência, com suas anotações e proveniência."""

    sample_id: str
    artifact: SampleArtifact
    width: int
    height: int
    split: Split
    provenance: AnnotationProvenance
    source_frame_id: str | None = None
    capture_conditions: tuple[str, ...] = field(default_factory=tuple)
    strata: tuple[str, ...] = field(default_factory=tuple)
    regions: tuple[RegionAnnotation, ...] = field(default_factory=tuple)
    relations: tuple[RelationAnnotation, ...] = field(default_factory=tuple)
    review_state: ReviewState = ReviewState.PENDING_REVIEW

    # Valida identidade, resolução, unicidade das anotações, coerência das
    # máscaras com a resolução da amostra, e ausência de relação pendurada.
    def __post_init__(self) -> None:
        """Valida identidade, geometria e integridade referencial da amostra."""
        if not self.sample_id.strip():
            raise ValueError("SampleAnnotation.sample_id must not be empty.")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"SampleAnnotation({self.sample_id!r}) must have a positive resolution.")

        annotation_ids = [region.annotation_id for region in self.regions]
        duplicated = sorted(name for name, count in Counter(annotation_ids).items() if count > 1)
        if duplicated:
            raise ValueError(
                f"SampleAnnotation({self.sample_id!r}) has duplicate annotation ids: {duplicated}."
            )
        relation_ids = [relation.relation_id for relation in self.relations]
        duplicated = sorted(name for name, count in Counter(relation_ids).items() if count > 1)
        if duplicated:
            raise ValueError(
                f"SampleAnnotation({self.sample_id!r}) has duplicate relation ids: {duplicated}."
            )

        for region in self.regions:
            if (region.mask.width, region.mask.height) != (self.width, self.height):
                raise ValueError(
                    f"RegionAnnotation({region.annotation_id!r}) mask resolution does not match "
                    f"sample {self.sample_id!r}."
                )
        known = set(annotation_ids)
        for relation in self.relations:
            for annotation_id, role in (
                (relation.subject_annotation_id, "subject_annotation_id"),
                (relation.object_annotation_id, "object_annotation_id"),
            ):
                if annotation_id not in known:
                    raise ValueError(
                        f"RelationAnnotation({relation.relation_id!r}).{role} references unknown "
                        f"annotation {annotation_id!r} in sample {self.sample_id!r}."
                    )

    # Filtra as regiões que participam do cálculo de métrica, deixando de
    # fora as marcadas como ignoradas. Existe para que cada consumidor não
    # replique (e eventualmente esqueça) essa regra.
    @property
    def scored_regions(self) -> tuple[RegionAnnotation, ...]:
        """As regiões que contam para métrica, excluindo as ignoradas."""
        return tuple(region for region in self.regions if not region.ignored)


# Raiz do manifest de referência: identidade, versão, política e amostras.
# Existe como o objeto único que a avaliação (#198/#199) carrega para saber
# o que medir e contra o quê.
@dataclass(frozen=True)
class ReferenceManifest:
    """O conjunto de referência anotado completo, com sua política e amostras."""

    reference_id: str
    dataset_id: str
    policy_uri: str
    samples: tuple[SampleAnnotation, ...]
    schema_version: str = REFERENCE_SCHEMA_VERSION
    created_at: str | None = None

    # Valida a versão de schema, a identidade do dataset e as duas regras
    # que só podem ser verificadas com o conjunto inteiro em mãos:
    # identidade duplicada de amostra e vazamento entre splits.
    def __post_init__(self) -> None:
        """Valida versão, identidade das amostras e ausência de vazamento de split."""
        for name in ("reference_id", "policy_uri"):
            if not getattr(self, name).strip():
                raise ValueError(f"ReferenceManifest.{name} must not be empty.")
        validate_dataset_name(self.dataset_id)
        if self.schema_version != REFERENCE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {REFERENCE_SCHEMA_VERSION!r}."
            )
        if not self.samples:
            raise ValueError("a reference manifest must declare at least one sample.")

        sample_ids = [sample.sample_id for sample in self.samples]
        duplicated = sorted(name for name, count in Counter(sample_ids).items() if count > 1)
        if duplicated:
            raise ValueError(f"reference manifest has duplicate sample ids: {duplicated}.")
        _reject_split_leakage(self.samples)

    # Filtra as amostras de um split. Usada pelos experimentos, que avaliam
    # cada split separadamente e nunca podem misturá-los por engano.
    def samples_in(self, split: Split) -> tuple[SampleAnnotation, ...]:
        """Retorna as amostras pertencentes a ``split``, na ordem do manifest."""
        return tuple(sample for sample in self.samples if sample.split is split)


# Recusa a mesma evidência aparecendo em mais de um split. Existe porque
# vazamento de split é o erro que invalida silenciosamente um resultado
# inteiro: ele não quebra nada, apenas infla a métrica. Chamada por
# ReferenceManifest.__post_init__ e por validate_reference.
def _reject_split_leakage(samples: tuple[SampleAnnotation, ...]) -> None:
    """Levanta se o mesmo artifact ou frame de origem aparecer em splits diferentes."""
    for label, key in (("artifact digest", "digest"), ("source frame", "source_frame_id")):
        splits_by_key: dict[str, set[str]] = {}
        for sample in samples:
            identity = sample.artifact.digest if key == "digest" else sample.source_frame_id
            if identity is None:
                continue
            splits_by_key.setdefault(identity, set()).add(sample.split.value)
        leaked = sorted(
            identity for identity, splits in splits_by_key.items() if len(splits) > 1
        )
        if leaked:
            raise ValueError(
                f"split leakage: the same {label} appears in more than one split: {leaked}."
            )


# Valida um manifest já construído sob as regras adicionais de publicação.
# Existe separada das invariantes de construção porque "pronto para gerar
# números publicáveis" é mais estrito que "estruturalmente válido": um
# conjunto em anotação continua sendo um manifest legítimo.
def validate_reference(
    manifest: ReferenceManifest,
    *,
    require_reviewed: bool = False,
    required_splits: tuple[Split, ...] = (Split.DEVELOPMENT, Split.CALIBRATION, Split.TEST),
) -> None:
    """Valida um manifest para uso em avaliação, com as regras de release opcionais.

    Argumentos:
        manifest: o conjunto de referência a validar.
        require_reviewed: exige que toda amostra esteja revisada.
        required_splits: splits que precisam ter ao menos uma amostra.
    Levanta:
        ValueError: se faltar split, ou se ``require_reviewed`` não for atendido.
    """
    missing = [split.value for split in required_splits if not manifest.samples_in(split)]
    if missing:
        raise ValueError(f"reference manifest has no samples for splits: {missing}.")
    if not require_reviewed:
        return
    unreviewed = sorted(
        sample.sample_id
        for sample in manifest.samples
        if sample.review_state is not ReviewState.REVIEWED
    )
    if unreviewed:
        raise ValueError(
            "reference manifest is not releasable: these samples have not been reviewed: "
            f"{unreviewed}."
        )
