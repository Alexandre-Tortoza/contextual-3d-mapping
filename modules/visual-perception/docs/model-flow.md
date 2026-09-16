# Model Dataflow

Este documento detalha onde cada backend participa do fluxo e evita a interpretação incorreta de que a pipeline é simplesmente `SAM -> DINO -> VLM -> CLIP`.

## Fluxo por modelo

```mermaid
flowchart TD
    RGB[RGB Frame]

    RGB --> SAMG[SAM3 Tracker]
    SAMG --> GM[generic masks]

    RGB --> CVLM[Qwen / Gemini: scene concept discovery]
    CVLM --> CONCEPTS[scene concepts]
    CONCEPTS --> SAMP[SAM3 PCS]
    RGB --> SAMP
    SAMP --> CM[concept-conditioned masks]

    GM --> REGIONS[ObservedRegion array]
    CM --> REGIONS

    RGB --> DINO[DINOv2]
    DINO --> PATCH[dense patch embeddings]
    PATCH --> DENSE[region dense visual evidence]

    RGB --> SCVLM[Qwen / Gemini: scene reasoning]
    SCVLM --> SCENE[SceneContext]

    REGIONS --> VIEWS[masked_subject + tight_crop + contextual_crop]
    VIEWS --> RVLM[Qwen / Gemini: region reasoning]
    SCENE --> RVLM
    RVLM --> HYP[PRIMARY + ALTERNATIVE hypotheses]

    VIEWS --> CIMG[CLIP image encoder]
    CIMG --> IEMB[visual embeddings]

    HYP --> CTXT[CLIP text encoder]
    CTXT --> TEMB[language embeddings]

    IEMB --> SUPPORT[independent visual-text support]
    TEMB --> SUPPORT

    DENSE -.->|não é enviado diretamente| RVLM

    HYP --> CLAIM[SemanticClaim]
    SUPPORT --> CLAIM
    CLAIM --> CAL[calibration]
    CAL --> REF[refinement]
    REF --> OUT[contextual semantic state]
```

## Responsabilidade de cada modelo

```mermaid
flowchart LR
    SAM[SAM3] --> WHERE[Onde existe uma região?]
    DINO[DINOv2] --> LOOK[Como patches e regiões se parecem visualmente?]
    VLM[Qwen / Gemini] --> MEAN[O que a cena ou região pode significar?]
    CLIP[CLIP] --> COMPAT[A evidência visual é compatível com a hipótese textual?]
```

## DINO e VLM são caminhos separados

```mermaid
flowchart TD
    RGB[RGB] --> DINO[DINOv2]
    DINO --> FM[Dense FeatureMap]
    FM --> DE[dense region evidence]

    RGB --> VIEW[RegionView pixels]
    SC[SceneContext] --> VLM[Qwen / Gemini]
    VIEW --> VLM
    VLM --> H[semantic hypotheses]

    DE -.->|não alimenta diretamente o reasoner| VLM

    VIEW --> CI[CLIP image]
    H --> CT[CLIP text]
    CI --> S[hypothesis support]
    CT --> S
```

DINO e CLIP podem ter a mesma dimensionalidade em uma configuração específica, mas vivem em espaços de representação diferentes e seus vetores não devem ser comparados diretamente.