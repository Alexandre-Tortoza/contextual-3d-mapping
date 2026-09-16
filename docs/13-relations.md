# 13. Relations

## Onde estamos na pipeline

```mermaid
flowchart LR
    A["ObservedRegion[] já reconciliadas"] --> G["Geometric relation generation"]
    A --> S["Semantic relation candidate selection"]
    C["2D masks + boxes"] --> G
    C --> S
    V["pair view + VLM"] --> S

    G --> GR["GEOMETRIC_2D CandidateRelation[]"]
    S --> SR["MODEL_INFERRED CandidateRelation[]"]

    GR --> ALL["CandidateRelation[]"]
    SR --> ALL
    ALL --> VO["VisualObservation"]
```

## Objetivo

Registrar relações candidatas entre regiões do mesmo frame sem promovê-las automaticamente a relações métricas 3D.

O contract atual distingue duas fontes:

```mermaid
flowchart TD
    REL["CandidateRelation"]

    REL --> G["RelationSource.GEOMETRIC_2D"]
    REL --> M["RelationSource.MODEL_INFERRED"]

    G --> GM["medida diretamente de masks / boxes"]
    M --> MM["inferida pelo reasoner multimodal"]

    GM --> CAND["continua sendo relação 2D candidata"]
    MM --> CAND
```

Nenhuma dessas arestas é automaticamente uma relação física verificada em 3D.

## Contract

```text
CandidateRelation
{
    relation_id
    subject_region_id
    predicate
    object_region_id
    confidence?
    source
    evidence[]
    provenance
}
```

A relação sempre referencia duas regiões canônicas diferentes e precisa carregar pelo menos uma evidência. Predicados são normalizados em `snake_case`.

## Vocabulário completo atual

A implementação atual possui dois vocabulários distintos.

```mermaid
flowchart TD
    R["CandidateRelation predicates"]

    R --> G["GEOMETRIC_2D"]
    G --> GO["overlaps"]
    G --> GC["contains"]
    G --> GN["near"]

    R --> M["MODEL_INFERRED"]
    M --> MP["part_of"]
    M --> MA["attached_to"]
    M --> MS["supported_by"]
    M --> MC["covers"]
    M --> MO["occludes"]
```

Esses são os predicados de relação emitidos pelo caminho canônico atual.

## Relações geométricas 2D

As relações geométricas são geradas diretamente de masks e boxes, sem VLM.

```mermaid
flowchart TD
    A["Region A mask + box"]
    B["Region B mask + box"]

    A --> IOU["mask IoU"]
    B --> IOU

    A --> CONT["containment ratio"]
    B --> CONT

    A --> GAP["box gap / proximity"]
    B --> GAP

    IOU -->|"> 0"| OVER["overlaps"]
    CONT -->|">= 0.90 em uma direção"| CONTAINS["contains"]
    GAP -->|"até 5 px e sem overlap"| NEAR["near"]
```

### `overlaps`

```mermaid
flowchart LR
    A["region-A"] -->|"mask IoU > 0"| R["overlaps"]
    R --> B["region-B"]
```

A confiança é o próprio IoU medido.

```text
confidence = mask_iou
```

Não significa que os dois elementos ocupam o mesmo volume no mundo; significa apenas que suas máscaras 2D se sobrepõem na imagem.

### `contains`

```mermaid
flowchart LR
    A["container region"] -->|"containment >= 0.90"| R["contains"]
    R --> B["contained region"]
```

A direção é explícita. O pipeline compara os dois sentidos e emite `contains` somente para a direção dominante.

A confiança é o containment ratio medido.

### `near`

```mermaid
flowchart LR
    A["region-A box"] --> GAP["gap <= 5 px"]
    B["region-B box"] --> GAP
    GAP --> R["near"]
```

`near` só é emitido quando não existe overlap. A confiança é uma proximidade normalizada:

```text
1.0
    boxes encostados

0.0
    exatamente no limite de 5 px
```

`near` é proximidade em imagem, não distância métrica 3D.

## Relações semânticas inferidas por modelo

Depois da reconciliação intra-frame, o pipeline pode selecionar pares para uma chamada multimodal. O reasoner recebe uma única view conjunta do par, mantendo a relação espacial entre os dois sujeitos.

```mermaid
flowchart TD
    A["Reconciled ObservedRegion A"]
    B["Reconciled ObservedRegion B"]

    A --> SEL["candidate selection"]
    B --> SEL

    SEL --> GEOM["containment + overlap + touching"]
    GEOM --> BUDGET["diversification + max_pairs"]

    BUDGET --> VIEW["pair view\ngreen subject + blue object"]
    VIEW --> VLM["Qwen / Gemini relation reasoning"]

    VLM --> PRED{"predicate"}

    PRED --> P1["part_of"]
    PRED --> P2["attached_to"]
    PRED --> P3["supported_by"]
    PRED --> P4["covers"]
    PRED --> P5["occludes"]
    PRED --> NONE["none"]
```

`none` é uma resposta válida e significa que nenhuma relação semântica do vocabulário é sustentada. Nesse caso nenhuma `CandidateRelation` é criada.

### `part_of`

```mermaid
flowchart LR
    PART["região que representa uma parte"] -->|"part_of"| WHOLE["região que representa o todo"]
```

Uso pretendido: composição de entidades e objetos quando uma região visual é semanticamente parte de outra.

Exemplo conceitual:

```text
handle --part_of--> door
```

### `attached_to`

```mermaid
flowchart LR
    A["region A"] -->|"attached_to"| B["region B"]
```

Representa uma conexão visual observável no frame em que uma entidade parece fisicamente presa a outra.

Exemplo conceitual:

```text
sign --attached_to--> panel
```

### `supported_by`

```mermaid
flowchart LR
    A["supported entity"] -->|"supported_by"| B["supporting surface/entity"]
```

Representa suporte visual observável. É um candidato útil para um futuro `scene-graph`, mas ainda não é verificação métrica de contato ou força em 3D.

Exemplo conceitual:

```text
box --supported_by--> shelf
```

### `covers`

```mermaid
flowchart LR
    A["covering region"] -->|"covers"| B["covered region"]
```

Expressa que uma região visual aparece cobrindo outra. É semanticamente diferente de `contains`: `contains` vem da geometria das máscaras, enquanto `covers` afirma uma interpretação da relação entre os conteúdos das regiões.

### `occludes`

```mermaid
flowchart LR
    A["foreground region"] -->|"occludes"| B["partially hidden region"]
```

Expressa oclusão visual inferida dentro do frame. Ela pode informar etapas downstream, mas não substitui o teste geométrico de visibilidade em `sensor-association`.

## Relações que NÃO pertencem ao vocabulário semântico atual

O reasoner não pode emitir qualquer string arbitrária como predicado. O vocabulário é fechado e versionado.

```mermaid
flowchart TD
    OUT["Predicados fora do vocabulário atual"]

    OUT --> A["above / below"]
    OUT --> B["behind / in_front_of"]
    OUT --> C["metric_distance"]
    OUT --> D["reachability / navigability relation"]
    OUT --> E["inter-frame relation"]
    OUT --> F["adjacent_to"]
    OUT --> G["inside"]

    F --> FN["já representado geometricamente por near"]
    G --> GN["inversa geométrica de contains"]

    A --> NEED3D["exige geometria / interpretação downstream"]
    B --> NEED3D
    C --> NEED3D
    D --> NEED3D
    E --> NEED3D
```

`above`, `below`, `behind` e `in_front_of` não são aceitos atualmente como relações semânticas inferidas porque a intenção do contract é não fingir geometria métrica 3D a partir de um único frame.

`adjacent_to` não existe no vocabulário semântico porque o canal geométrico já responde essa pergunta como `near`.

`inside` não existe porque é a relação inversa de `contains`; mantê-la no VLM duplicaria uma medida que a geometria já calcula diretamente.

## Como pares são selecionados

Perguntar ao VLM sobre todos os pares é quadrático.

```mermaid
flowchart TD
    ALL["todos os pares N*(N-1)/2"]
    ALL --> GEOM["priorizar containment / overlap / touching"]
    GEOM --> GROUP["remover pares do mesmo entity group"]
    GROUP --> SAME["remover conceitos idênticos quando configurado"]
    SAME --> PERREGION["max_pairs_per_region"]
    PERREGION --> GLOBAL["max_pairs"]
    GLOBAL --> VLM["relation reasoning"]
```

A direção do par é escolhida a partir da geometria, usando a região com maior containment como sujeito. Isso evita perguntar duas vezes sobre o mesmo par em sentidos arbitrários.

## Proveniência e confiança

```mermaid
flowchart TD
    G["GEOMETRIC_2D"] --> GC["confidence = medida geométrica"]
    GC --> G1["overlaps -> IoU"]
    GC --> G2["contains -> containment ratio"]
    GC --> G3["near -> normalized proximity"]

    M["MODEL_INFERRED"] --> MC["confidence = score informado pelo producer"]
    MC --> NONE["pode ser None"]
```

Uma relação inferida por modelo sem score continua válida com `confidence=None`; o pipeline não inventa `1.0` ou outro número.

As fontes coexistem. Uma relação geométrica não é sobrescrita por uma relação inferida por modelo e vice-versa.

## Relação com `reconciliation`

A inferência semântica acontece depois da reconciliação intra-frame.

```mermaid
flowchart LR
    REG["region semantics"] --> REC["reconciliation"]
    REC --> ENTITY["ContextualEntityHypothesis[]"]
    REC --> SEMREL["semantic relation candidate selection"]

    ENTITY --> FILTER["pares internos ao mesmo grupo são excluídos"]
    FILTER --> SEMREL
```

Se duas regiões já foram reconciliadas como membros da mesma hipótese de entidade, o estágio de relações não tenta descrever novamente esse vínculo como uma aresta semântica.

## Relações 2D versus relações 3D futuras

```mermaid
flowchart TD
    R2D["CandidateRelation 2D"]
    R2D --> ASSOC["sensor-association"]
    ASSOC --> GEO["geometry support"]

    GEO -.-> SG["future scene-graph"]
    R2D -.-> SG

    SG -.-> VERIFIED["persistent relation with 3D support + provenance"]
```

O sistema deverá continuar distinguindo:

```text
relação observada ou inferida em 2D
relação geometricamente suportada em 3D
relação inferida por contexto de alto nível
```

Uma `CandidateRelation` nunca deve ser promovida automaticamente a fato espacial persistente apenas porque o VLM retornou um predicado.

## Saída

```mermaid
flowchart LR
    G["geometric CandidateRelation[]"] --> ALL["VisualObservation.relations"]
    M["model-inferred CandidateRelation[]"] --> ALL

    ALL --> NEXT["downstream 2D->3D association / future graph reasoning"]
```

Cada relação preserva os IDs das duas regiões, fonte, evidência, proveniência e confiança quando disponível.

## Próxima leitura

- [Fluxo completo do repositório](./system-flow.md)
- [14. VisualObservation](./14-visual-observation.md)
- [12. Reconciliation intra-frame](./12-reconciliation.md)
- [17. Sensor Association](./17-sensor-association.md)
- [Fluxo semântico interno de `visual-perception`](../modules/visual-perception/docs/semantic-flow.md)
