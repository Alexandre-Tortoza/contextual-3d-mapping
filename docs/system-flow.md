# Fluxo do Sistema

Este documento mostra o fluxo de nível de repositório entre módulos e aplicações. Ele representa intencionalmente composição e movimento de dados entre fronteiras, não algoritmos internos.

Setas entre módulos representam troca através de contracts compatíveis, não dependências diretas de implementação.

```mermaid
flowchart LR
    D[Datasets] --> A[adapters]
    S[Sensores ao vivo ou gravados] --> A

    A --> SE[state-estimation]
    A --> VP[visual-perception]

    SE -->|pose / trajectory| GM[geometric-map]
    SE -->|LiDAR corrigido por movimento| GM
    SE -->|LiDAR corrigido por movimento| PR[point-representation]
    SE -->|pose / trajectory| SA[sensor-association]

    GM -->|refs de geometria persistente| SA
    VP --> SA
    PR --> SA

    SA --> SF[semantic-fusion]
    SF --> SMAP[semantic-map]
    GM -->|refs de geometria| SMAP

    SMAP --> SMEM[semantic-memory]
    SMAP --> SG[scene-graph]

    SMEM --> CR[context-reasoning]
    SG --> CR

    SMEM --> QE[query-engine]
    SG --> QE
    CR --> QE

    GM --> MR[apps/mapping-runtime]
    SMAP --> MR
    SMEM --> MR
    SG --> MR

    QE --> EX[apps/map-explorer]
    GM --> EX
    SMAP --> EX

    QE --> CLI[apps/cli]

    C[contracts] -. interfaces compartilhadas .-> A
    C -. interfaces compartilhadas .-> SE
    C -. interfaces compartilhadas .-> GM
    C -. interfaces compartilhadas .-> VP
    C -. interfaces compartilhadas .-> PR
    C -. interfaces compartilhadas .-> SA
    C -. interfaces compartilhadas .-> SF
    C -. interfaces compartilhadas .-> SMAP
    C -. interfaces compartilhadas .-> SMEM
    C -. interfaces compartilhadas .-> SG
    C -. interfaces compartilhadas .-> CR
    C -. interfaces compartilhadas .-> QE
```

## Fluxo de mapeamento

O fluxo de mapeamento converte observações de sensor ou dataset em estado geométrico, semântico e contextual persistente.

```text
observações de entrada
    -> state estimation
    -> geometria persistente
    -> representações aprendidas e visuais
    -> sensor association
    -> semantic fusion
    -> semantic map
    -> semantic memory / scene graph
    -> raciocínio contextual e índices
```

`mapping-runtime` é o ponto de entrada de composição para esse fluxo. Ele não substitui nenhum módulo e não possui seus algoritmos.

## Fluxo de consulta e exploração

Depois que um mapa existe, a interação em tempo de consulta não exige reexecutar a state estimation. Aplicações consomem o estado de mapa persistente e as interfaces públicas de consulta.

```mermaid
flowchart LR
    U[Consulta do usuário] --> EX[map-explorer]
    EX --> QE[query-engine]
    QE --> SMEM[semantic-memory]
    QE --> SG[scene-graph]
    QE --> CR[context-reasoning]
    QE --> R[resultados da consulta]
    R --> EX
    GM[geometric-map] --> EX
    SMAP[semantic-map] --> EX
    OBS[observações / evidência] --> EX
```

Um resultado pode identificar uma entidade, região, posição, referência de geometria, relação, observação, item de evidência ou registro de proveniência. A aplicação pode então focar a região 3D correspondente e expor as observações que sustentam o resultado.

## Fluxo de persistência

A persistência é uma fronteira transversal (cross-cutting), não um estágio de pesquisa sequencial. Contracts públicos de armazenamento permitem que estado de mapa, observações, evidência e índices sobrevivam ao término do processo e sejam reabertos por outra aplicação.

Formatos concretos de armazenamento e tecnologias de banco de dados permanecem adapters.

## Interpretação

`evaluation/`, `experiments/` e `tests/` são intencionalmente não representados como estágios sequenciais. Eles são consumidores transversais que podem exercitar módulos individuais ou composições completas independentemente.

Um workflow pode substituir, isolar ou omitir estágios quando seus contracts permitirem essa composição. Poses de ground-truth fornecidas por dataset, por exemplo, podem substituir um state estimator ao vivo em um experimento sem alterar os contracts públicos downstream.
