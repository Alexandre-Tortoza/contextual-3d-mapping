# Semantic Reasoning Flow

Este documento começa na região cuja geometria já foi consolidada. A partir do geometry freeze, os estágios semânticos acrescentam evidência, claims, sinais, grupos e relações, sem alterar `mask`, `bbox` ou `region_id`.

## Fluxo semântico completo

```mermaid
flowchart TD
    R[ObservedRegion after geometry freeze]

    R --> V[build RegionViews]
    V --> MS[masked_subject]
    V --> TC[tight_crop]
    V --> CC[contextual_crop]

    MS --> CI[CLIP image encoder]
    TC --> CI
    CC --> CI
    CI --> VE[VisualEmbedding array]

    R --> DINO[DINO region evidence from dense FeatureMap]

    RGB[RGB frame] --> SCENE[Qwen / Gemini scene analysis]
    MASKS[valid area - ego] --> SCENE
    SCENE --> SC[SceneContext]

    R --> PRIOR{ScenePrior available?}
    VE --> PRIOR
    PRIOR -->|sim| PM[match_scene_prior]
    PM --> PA[PriorAssignment]
    PRIOR -->|não| NP[no prior]

    MS --> REQ[RegionReasoningRequest]
    TC --> REQ
    CC --> REQ
    SC --> REQ
    PA --> REQ
    NP --> REQ

    REQ --> VLM[Qwen / Gemini region reasoning]
    VLM --> P[PRIMARY]
    VLM --> A[ALTERNATIVE]
    VLM --> C[category]
    VLM --> K[region_kind]
    VLM --> AT[attributes]
    P --> CLAIM[SemanticClaim array]
    A --> CLAIM
    C --> CLAIM
    K --> CLAIM
    AT --> CLAIM

    CLAIM --> TXT[versioned hypothesis text / template]
    TXT --> CT[CLIP text encoder]
    CT --> LE[LanguageEmbedding array]

    VE --> HS[attach_hypothesis_signals]
    LE --> HS
    HS --> SUPPORT[supports / contradicts / indistinguishable / unavailable]

    SUPPORT --> CAL[semantic calibration]
    SC --> CAL
    CAL --> STATE[calibrated observation state]

    STATE --> REF{selective refinement needed?}
    REF -->|não| STABLE[stable claims]
    REF -->|sim| VLM2[additional VLM reasoning]
    VLM2 --> NH[new / revised hypotheses]
    NH --> HS2[repeat independent support]
    HS2 --> CAL2[repeat calibration]
    CAL2 --> STABLE

    STABLE --> GR[geometric relations]
    GR --> REC[intra-frame reconciliation]
    REC --> GROUP[ContextualEntityHypothesis + ReconciliationRecord]
    GROUP --> SR[semantic relation inference]
    SC --> SR
    SR --> REL[semantic relations]

    REL --> PUB[contextual publication]
    PUB --> STRUCT{generic structural surface only?}
    STRUCT -->|não| REG[observation.regions]
    STRUCT -->|sim| SCTX[observation.structural_context]
    SCTX --> SUPP[SuppressedRegion]

    REG --> OBS[final VisualObservation]
    SCTX --> OBS
    OBS --> AUDIT[final audit]
    AUDIT --> OUT[AuditResult + published observation]
```

## Qwen propõe, CLIP mede suporte

```mermaid
flowchart TD
    VIEWS[RegionView pixels] --> Q[Qwen / Gemini]
    SC[SceneContext] --> Q
    Q --> H1[PRIMARY: hypothesis A]
    Q --> H2[ALTERNATIVE: hypothesis B]

    VIEWS --> CI[CLIP image encoder]
    H1 --> CT[CLIP text encoder]
    H2 --> CT

    CI --> SIM[visual-text similarity + relative margin]
    CT --> SIM
    SIM --> SIG[SupportSignal]

    SIG --> NOTE[CLIP não troca a label sozinho e o score não é probabilidade calibrada]
```

## Refinamento preserva os mesmos crivos

```mermaid
flowchart LR
    C[calibrated claims] --> R{refinement reason?}
    R -->|não| O[keep]
    R -->|sim| V[additional VLM reasoning]
    V --> N[new hypothesis]
    N --> S[independent support]
    S --> C2[calibration]
    C2 --> O
    V --> H[append-only RefinementStep history]
```

## Publicação contextual

```mermaid
flowchart TD
    R[reconciled regions + relations] --> P[partition_observation]
    P --> Q{asserted identity only generic structural surface?}
    Q -->|não| PUB[keep in observation.regions]
    Q -->|sim| STR[move intact to observation.structural_context]
    STR --> REC[record SuppressedRegion + reason]
    PUB --> OBS[VisualObservation]
    STR --> OBS
    OBS --> A[final audit sees the complete observation]
```

A publicação decide o que é exposto como região contextual relevante. Ela não apaga a observação estrutural e não redefine sua geometria.