# Fluxo completo do contextual-3d-mapping

Este documento descreve a pipeline **completa do repositório**, não apenas o módulo `visual-perception`. Ele mostra como dados, contracts, referências, geometria, evidência visual, semântica, proveniência e aplicações se relacionam do ingresso dos sensores até o mapa contextual consolidado.

A regra desta página é distinguir explicitamente:

```text
fluxo implementado
    !=
fluxo opcional / lateral
    !=
capacidade planejada
```

Linhas contínuas representam relações implementadas ou contracts materializados. Linhas tracejadas representam capacidades planejadas ou integrações futuras ainda não materializadas como fluxo canônico.

## 1. Pipeline canônica completa

```mermaid
flowchart TD

    %% =========================================================
    %% SOURCES
    %% =========================================================

    subgraph SRC["FONTES EXTERNAS"]
        LIVE["Sensores ao vivo\nRGB + LiDAR + IMU"]
        BAG["ROS bag / sessão gravada"]
        DATASET["Dataset adapter"]
        FIXTURE["Fixture / benchmark / teste"]
        CALIB_FILE["Artifact de calibração\nintrínseca + extrínseca"]
    end

    %% =========================================================
    %% ADAPTERS
    %% =========================================================

    subgraph ADP["ADAPTERS DE REPOSITÓRIO"]
        ROS2["adapters/ros2\ntradução ROS 2 -> contracts"]
        DSADP["adapters/datasets\nnormalização de dataset"]
        STORAGE["adapters/map-storage\npersistência / artifacts"]
    end

    LIVE --> ROS2
    BAG --> ROS2
    DATASET --> DSADP

    %% =========================================================
    %% SHARED CONTRACTS
    %% =========================================================

    subgraph CONTRACTS["CONTRACTS COMPARTILHADOS"]
        OBSREF["ObservationReference"]
        TIME["timestamps / clock_id"]
        SPATIAL["frames / transforms"]
        RUN["run provenance"]
        MAPCONTRACT["map identifiers / map contracts"]
    end

    ROS2 --> OBSREF
    ROS2 --> TIME
    ROS2 --> SPATIAL
    DSADP --> OBSREF
    DSADP --> TIME
    DSADP --> SPATIAL
    FIXTURE --> OBSREF

    %% =========================================================
    %% COMPOSITION ROOT
    %% =========================================================

    subgraph RUNTIME["apps/mapping-runtime  |  COMPOSITION ROOT"]
        ORCH["wiring + configuração\nordem de execução\nlifecycle"]
        WINDOW["bag-window / keyframe identity"]
        RUNCTX["run / segment identity"]
    end

    OBSREF --> ORCH
    TIME --> ORCH
    SPATIAL --> ORCH
    BAG --> WINDOW
    WINDOW --> ORCH
    RUN --> RUNCTX
    RUNCTX --> ORCH

    %% =========================================================
    %% RGB BRANCH
    %% =========================================================

    subgraph RGBPATH["RAMO RGB  |  visual-perception"]
        RGB["ImageObservation + ImagePayload"]
        AREAS["ImageAreaMasks\nvalid area + ego"]
        SCONCEPT["Scene Concept Discovery\nQwen / Gemini"]
        CGROUND["Concept Grounding\nSAM3 PCS"]
        GDISC["Generic Region Discovery\nSAM3 tracker + tiling"]
        PFILTER["Proposal Filtering"]
        RMERGE["Cross-scale Merge"]
        FREEZE["GEOMETRY FREEZE\nmask + bbox + region_id"]
        EVID["Multi-context Evidence\nRegionView + DINO + CLIP image"]
        SCENE["SceneContext\nQwen / Gemini"]
        PRIOR["Optional temporal prior"]
        RSEM["Region Semantics\nprimary + alternatives"]
        CLIPTXT["CLIP text embeddings"]
        SUPPORT["Independent hypothesis support"]
        CAL["Semantic calibration"]
        REFINE["Selective refinement"]
        GREL["Geometric relations"]
        RECON["Intra-frame reconciliation"]
        SREL["Semantic relations"]
        PUB["Contextual publication"]
        VAUDIT["Final audit"]
        VOBS["VisualObservation"]
    end

    ORCH --> RGB
    RGB --> AREAS
    RGB --> SCONCEPT
    AREAS --> SCONCEPT
    SCONCEPT --> CGROUND
    RGB --> GDISC
    CGROUND --> PFILTER
    GDISC --> PFILTER
    AREAS --> PFILTER
    PFILTER --> RMERGE
    RMERGE --> FREEZE
    FREEZE --> EVID
    RGB --> EVID
    RGB --> SCENE
    AREAS --> SCENE
    FREEZE --> PRIOR
    EVID --> PRIOR
    FREEZE --> RSEM
    EVID --> RSEM
    SCENE --> RSEM
    PRIOR --> RSEM
    RSEM --> CLIPTXT
    EVID --> SUPPORT
    CLIPTXT --> SUPPORT
    SUPPORT --> CAL
    SCENE --> CAL
    CAL --> REFINE
    REFINE --> GREL
    GREL --> RECON
    RECON --> SREL
    SCENE --> SREL
    SREL --> PUB
    PUB --> VAUDIT
    VAUDIT --> VOBS

    %% =========================================================
    %% LIDAR / MOTION BRANCH
    %% =========================================================

    subgraph MOTION["RAMO LiDAR + IMU  |  state-estimation"]
        LIDAR["LiDAR observation"]
        IMU["IMU observation"]
        EST["state-estimation\nFAST-LIO integration / contracts"]
        STATE["StateEstimate / trajectory / pose"]
        MCLOUD["motion-corrected LiDAR"]
    end

    ORCH --> LIDAR
    ORCH --> IMU
    LIDAR --> EST
    IMU --> EST
    EST --> STATE
    EST --> MCLOUD

    %% =========================================================
    %% GEOMETRIC MAP
    %% =========================================================

    subgraph GMAP["geometric-map  |  GEOMETRIA PERSISTENTE"]
        GPERSIST["persist / transform geometry"]
        GPOINT["GeometryPoint"]
        GREF["GeometryReference\nmap_id + geometry_id"]
        BOUNDS["Bounds3D / spatial access"]
    end

    STATE --> GPERSIST
    MCLOUD --> GPERSIST
    MAPCONTRACT --> GPERSIST
    GPERSIST --> GPOINT
    GPERSIST --> GREF
    GPERSIST --> BOUNDS

    %% =========================================================
    %% POINT REPRESENTATION SIDE PATH
    %% =========================================================

    subgraph PREP["point-representation  |  RAMO OPCIONAL"]
        PTRANS["deterministic point transforms"]
        PENC["PointEncoder port"]
        PEMB["PointEmbedding / 3D representation"]
    end

    GPOINT --> PTRANS
    PTRANS --> PENC
    PENC --> PEMB

    %% =========================================================
    %% CALIBRATION + POSE
    %% =========================================================

    subgraph CALPOSE["POSE + CALIBRAÇÃO"]
        CALIB["CameraLidarCalibration\nmodel + intrinsics + lidar_to_camera"]
        POSE["pose map/world -> camera\nat RGB timestamp"]
    end

    CALIB_FILE --> CALIB
    STATE --> POSE

    %% =========================================================
    %% SENSOR ASSOCIATION
    %% =========================================================

    subgraph ASSOC["sensor-association  |  2D -> 3D"]
        PROJ["transform GeometryPoint -> camera"]
        CAMERA["camera model projection\npinhole / fisheye / MEI"]
        SUPPORT3D["valid optical support"]
        OCC["visibility / occlusion"]
        MASKMEM["ObservedRegion mask membership"]
        STATUS["AssociationStatus\nassociated / behind / outside / occluded"]
        PVA["PointVisualAssociation"]
    end

    GPOINT --> PROJ
    POSE --> PROJ
    CALIB --> PROJ
    PROJ --> CAMERA
    CAMERA --> SUPPORT3D
    SUPPORT3D --> OCC
    VOBS --> MASKMEM
    OCC --> MASKMEM
    MASKMEM --> STATUS
    GREF --> PVA
    STATUS --> PVA
    VOBS --> PVA
    CALIB --> PVA

    %% =========================================================
    %% SEMANTIC CONTRIBUTIONS + FUSION
    %% =========================================================

    subgraph FUSION["semantic-fusion  |  MULTI-VIEW + 3D"]
        CONTRIB["SemanticContribution\nper observation / region / point"]
        GROUP["group by GeometryReference"]
        RANK["quality ranking\ncalibrated_confidence > confidence > visual_support > region_quality"]
        AGREEMENT["multi-keyframe agreement"]
        SPATIALSUP["3D spatial support"]
        FUSED["FusedPointContext"]
    end

    PVA --> CONTRIB
    CONTRIB --> GROUP
    GREF --> GROUP
    GROUP --> RANK
    GROUP --> AGREEMENT
    BOUNDS --> SPATIALSUP
    GPOINT --> SPATIALSUP
    RANK --> FUSED
    AGREEMENT --> FUSED
    SPATIALSUP --> FUSED
    PEMB -. "canal futuro/independente" .-> FUSED

    %% =========================================================
    %% CONTEXT RUN
    %% =========================================================

    subgraph CONTEXTRUN["RUN CONTEXTUAL PUBLICADA"]
        CTX["context.json / contextual run"]
        PRUN["PublishedContextRun"]
        GFINGER["geometry_fingerprint"]
    end

    VOBS --> CTX
    PVA --> CTX
    FUSED --> CTX
    STATE --> CTX
    CALIB --> CTX
    GREF --> CTX
    RUNCTX --> CTX
    CTX --> PRUN
    GPOINT --> GFINGER
    GFINGER --> PRUN

    %% =========================================================
    %% SEMANTIC MAP
    %% =========================================================

    subgraph SMAP["semantic-map  |  CONSOLIDAÇÃO ENTRE RUNS"]
        COMPAT{"same map_id + map_frame +\ngeometry_fingerprint?"}
        CONSOL["consolidate_context_runs()"]
        CMAP["ConsolidatedContextMap"]
    end

    PRUN --> COMPAT
    COMPAT -->|"sim"| CONSOL
    COMPAT -->|"não"| REJECTRUN["reject incompatible run"]
    CONSOL --> CMAP

    %% =========================================================
    %% STORAGE + APPLICATIONS
    %% =========================================================

    CTX --> STORAGE
    CMAP --> STORAGE

    subgraph APPS["APPLICATIONS"]
        CLI["apps/cli\nrun / compare / publish"]
        EXPLORER["apps/map-explorer\n3D view + evidence inspection"]
    end

    ORCH --> CLI
    CLI --> STORAGE
    STORAGE --> EXPLORER

    %% =========================================================
    %% PLANNED CAPABILITIES
    %% =========================================================

    subgraph FUTURE["CAPACIDADES PLANEJADAS"]
        MEMORY["semantic-memory"]
        GRAPH["scene-graph"]
        CREASON["context-reasoning"]
        QUERY["query-engine"]
    end

    CMAP -.-> MEMORY
    CMAP -.-> GRAPH
    MEMORY -.-> CREASON
    GRAPH -.-> CREASON
    CREASON -.-> QUERY
    QUERY -.-> EXPLORER
    QUERY -.-> CLI
```

## 2. Relação de ownership entre módulos

```mermaid
flowchart LR
    VP["visual-perception"] -->|"dono de pixels, masks, RegionView, claims 2D"| VO["VisualObservation"]

    SE["state-estimation"] -->|"dono de pose, trajetória e contexto de movimento"| STATE["StateEstimate"]

    GM["geometric-map"] -->|"dono de XYZ persistente e identidade geométrica"| GEO["GeometryPoint / GeometryReference"]

    PR["point-representation"] -->|"dono de representação aprendida 3D, não da geometria"| PE["PointEmbedding"]

    SA["sensor-association"] -->|"dono da ligação observação 2D <-> ponto 3D"| PVA["PointVisualAssociation"]

    SF["semantic-fusion"] -->|"dono da fusão multi-view por geometria"| FUSED["FusedPointContext"]

    SM["semantic-map"] -->|"dono da consolidação entre runs compatíveis"| CMAP["ConsolidatedContextMap"]

    VO --> SA
    STATE --> SA
    GEO --> SA
    PVA --> SF
    GEO --> SF
    FUSED --> SM
```

A consequência dessa separação é importante:

- `visual-perception` não cria XYZ persistente;
- `geometric-map` não decide labels;
- `sensor-association` não pergunta novamente ao VLM o que existe no pixel;
- `semantic-fusion` não reprojeta pontos e não refaz percepção 2D;
- `semantic-map` não registra mapas diferentes nem resolve alinhamento espacial entre geometrias incompatíveis;
- `mapping-runtime` compõe capacidades, mas não deve implementar a lógica científica pertencente a um módulo.

## 3. Relação temporal e de identidade

A pipeline depende de identidade e tempo preservados desde a fonte. A relação completa é:

```mermaid
flowchart TD
    BAG["bag / dataset / live stream"]
    REC["recording_id"]
    OBS["ObservationReference"]
    TREC["recording timestamp"]
    THEAD["message/header timestamp"]
    CLOCK["clock_id"]

    BAG --> REC
    BAG --> TREC
    BAG --> THEAD
    THEAD --> CLOCK
    REC --> OBS
    THEAD --> OBS

    OBS --> RGB["ImageObservation"]
    OBS --> LIDAR["LiDAR observation"]
    OBS --> IMU["IMU observation"]

    LIDAR --> EST["state-estimation"]
    IMU --> EST
    EST --> TRAJ["trajectory"]

    RGB --> KEYFRAME["selected RGB keyframe"]
    KEYFRAME --> TS["exact RGB timestamp"]
    TRAJ --> INTERP["pose sampling / interpolation"]
    TS --> INTERP

    INTERP --> POSE["pose at RGB timestamp"]
    POSE --> ASSOC["sensor-association"]

    OBS --> PROV["provenance chain"]
    ASSOC --> PROV
```

No `corridor-02`, o runtime preserva separadamente o tempo de gravação do bag e o timestamp de header. A pose usada para associação deve corresponder ao instante RGB, e o runtime rejeita extrapolação ou gaps temporais acima da política configurada.

## 4. Relação geométrica entre LiDAR, mapa e câmera

```mermaid
flowchart LR
    PL["P_lidar"]
    STATE["StateEstimate / trajectory"]
    PMAP["P_map / GeometryPoint"]
    POSE["T_camera_map no timestamp RGB"]
    PCAM["P_camera"]
    CAL["CameraLidarCalibration"]
    PIXEL["pixel u,v"]
    VALID["valid optical support"]
    OCC["visibility / occlusion"]
    MASK["ObservedRegion mask"]
    PVA["PointVisualAssociation"]

    PL -->|"motion correction + map transform"| PMAP
    STATE --> PMAP
    PMAP -->|"map/world -> camera"| PCAM
    POSE --> PCAM
    PCAM -->|"camera model + intrinsics/distortion"| PIXEL
    CAL --> PIXEL
    PIXEL --> VALID
    VALID --> OCC
    OCC --> MASK
    MASK --> PVA
```

Essa relação é a fronteira onde um erro de pose, calibração ou visibilidade pode parecer um erro semântico. Por isso os diagnósticos mantêm esses canais separados.

## 5. Relação completa da evidência visual

```mermaid
flowchart TD
    RGB["RGB"]

    RGB --> SAM["SAM3 generic discovery"]
    RGB --> CVLM["SceneConceptDiscoverer"]
    CVLM --> CONCEPT["scene concepts"]
    CONCEPT --> PCS["SAM3 PCS"]
    RGB --> PCS

    SAM --> PROPOSALS["RegionProposal[]"]
    PCS --> PROPOSALS
    PROPOSALS --> FILTER["filter"]
    FILTER --> MERGE["merge"]
    MERGE --> REGION["ObservedRegion geometry"]

    REGION --> VIEW["RegionView[]"]
    RGB --> VIEW

    RGB --> DINO["DINOv2 FeatureMap"]
    DINO --> DENSE["mask-aware dense evidence"]
    REGION --> DENSE

    VIEW --> CLIPIMG["CLIP image encoder"]
    CLIPIMG --> VEMB["VisualEmbedding[]"]

    RGB --> SCENE["Qwen/Gemini scene reasoning"]
    SCENE --> SCTX["SceneContext"]

    VIEW --> RVLM["Qwen/Gemini region reasoning"]
    SCTX --> RVLM
    RVLM --> CLAIM["SemanticClaim[]\nPRIMARY + ALTERNATIVE"]

    CLAIM --> CLIPTXT["CLIP text encoder"]
    CLIPTXT --> LEMB["LanguageEmbedding[]"]

    VEMB --> SUPPORT["hypothesis support"]
    LEMB --> SUPPORT
    SUPPORT --> CAL["calibration"]
    CAL --> REFINE["selective refinement"]
    REFINE --> REL["relations + reconciliation"]
    REL --> PUB["contextual publication"]
    PUB --> VOBS["VisualObservation"]

    DENSE -. "não entra diretamente no VLM" .-> RVLM
```

Os canais têm funções diferentes:

```mermaid
flowchart LR
    SAM["SAM3"] -->|"onde existe uma região?"| GEO["2D region geometry"]
    DINO["DINOv2"] -->|"como patches/regiões se parecem?"| DENSE["dense visual representation"]
    VLM["Qwen / Gemini"] -->|"o que a cena/região pode significar?"| CLAIM["semantic hypotheses"]
    CLIP["CLIP"] -->|"a evidência visual combina com o texto?"| SUPPORT["independent support"]
```

## 6. Relação entre `VisualObservation` e geometria persistente

```mermaid
flowchart TD
    VO["VisualObservation"]
    REGION["ObservedRegion\nmask + claims + evidence"]
    GP["GeometryPoint"]
    GR["GeometryReference"]
    POSE["pose"]
    CAL["CameraLidarCalibration"]

    VO --> REGION
    GP --> GR

    REGION --> ASSOC["sensor-association"]
    GP --> ASSOC
    POSE --> ASSOC
    CAL --> ASSOC

    ASSOC --> PVA["PointVisualAssociation"]

    PVA --> LINK["explicit relation"]

    LINK --> L1["GeometryReference"]
    LINK --> L2["RGB ObservationReference"]
    LINK --> L3["LiDAR ObservationReference"]
    LINK --> L4["pixel"]
    LINK --> L5["region_id"]
    LINK --> L6["label / claim evidence"]
    LINK --> L7["feature_reference"]
    LINK --> L8["calibration"]
    LINK --> L9["AssociationStatus"]
```

`PointVisualAssociation` não transforma uma label 2D em verdade 3D. Ela registra que uma observação visual específica alcançou uma geometria específica sob uma pose, uma calibração e uma decisão de visibilidade específicas.

## 7. Relação multi-frame e fusão semântica

```mermaid
flowchart TD
    GR["GeometryReference X"]

    A["frame A / region A\ndoor"]
    B["frame B / region B\ndoor"]
    C["frame C / region C\npanel"]

    A --> CA["SemanticContribution A"]
    B --> CB["SemanticContribution B"]
    C --> CC["SemanticContribution C"]

    GR --> CA
    GR --> CB
    GR --> CC

    CA --> FUSION["semantic-fusion"]
    CB --> FUSION
    CC --> FUSION

    FUSION --> RANK["quality ranking"]
    FUSION --> AGREE["agreement"]
    FUSION --> SPATIAL["3D neighborhood support"]

    RANK --> FUSED["FusedPointContext"]
    AGREE --> FUSED
    SPATIAL --> FUSED

    FUSED --> PRIMARY["primary hypothesis"]
    FUSED --> CONTRIBUTORS["all contributions preserved"]
    FUSED --> QUALITY["agreement / support diagnostics"]
```

A fusão agrupa contribuições pela mesma `GeometryReference`; não sobrescreve cada observação conforme novos frames chegam.

## 8. Relação entre runs e `semantic-map`

```mermaid
flowchart TD
    R1["PublishedContextRun A"]
    R2["PublishedContextRun B"]
    R3["PublishedContextRun C"]

    R1 --> CHECK["compatibility gate"]
    R2 --> CHECK
    R3 --> CHECK

    CHECK --> MID["same map_id"]
    CHECK --> FRAME["same map_frame"]
    CHECK --> FP["same geometry_fingerprint"]

    MID --> OK{"compatible?"}
    FRAME --> OK
    FP --> OK

    OK -->|"não"| REJECT["do not consolidate"]
    OK -->|"sim"| CONSOL["consolidate_context_runs"]

    CONSOL --> VOTES["max one vote per point per run"]
    VOTES --> LABELGROUP["group textual variants"]
    LABELGROUP --> MAJ{"strict majority?"}

    MAJ -->|"sim"| WIN["majority result"]
    MAJ -->|"não"| TIE["quality tiebreak\nsemantic-fusion ranking"]

    WIN --> CMAP["ConsolidatedContextMap"]
    TIE --> CMAP
```

`semantic-map` consolida runs publicadas sobre **a mesma geometria**. Ele não é um módulo de registro espacial entre mapas independentes.

## 9. Relação de proveniência ponta a ponta

```mermaid
flowchart LR
    SOURCE["sensor / bag / dataset"]
    OBS["ObservationReference"]
    REGION["region_id"]
    CLAIM["SemanticClaim"]
    GROUND["grounding / mask"]
    GP["GeometryReference"]
    PVA["PointVisualAssociation"]
    CONTRIB["SemanticContribution"]
    FUSED["FusedPointContext"]
    RUN["PublishedContextRun"]
    CMAP["ConsolidatedContextMap"]

    SOURCE --> OBS
    OBS --> REGION
    REGION --> CLAIM
    REGION --> GROUND
    OBS --> PVA
    GROUND --> PVA
    GP --> PVA
    CLAIM --> PVA
    PVA --> CONTRIB
    CONTRIB --> FUSED
    FUSED --> RUN
    RUN --> CMAP
```

A cadeia deve permitir voltar de um ponto consolidado para:

```mermaid
flowchart TD
    FINAL["Consolidated semantic state"]
    FINAL --> RUN["source run"]
    RUN --> POINT["GeometryReference"]
    POINT --> CONTRIBUTION["SemanticContribution"]
    CONTRIBUTION --> ASSOC["PointVisualAssociation"]
    ASSOC --> FRAME["RGB / LiDAR observations"]
    ASSOC --> REGION["region_id + grounding"]
    REGION --> CLAIM["claim + alternatives + support"]
    ASSOC --> POSE["pose sampling"]
    ASSOC --> CAL["calibration"]
    CLAIM --> MODEL["models / prompts / config"]
```

## 10. Relação com armazenamento e aplicações

```mermaid
flowchart LR
    RUNTIME["mapping-runtime"] --> ART["contextual run artifacts"]
    SMAP["semantic-map"] --> CONS["ConsolidatedContextMap"]

    ART --> STORAGE["map-storage / artifact storage"]
    CONS --> STORAGE

    CLI["apps/cli"] -->|"run / save / compare / publish"| RUNTIME
    CLI --> STORAGE

    STORAGE --> EXPLORER["apps/map-explorer"]
    EXPLORER --> VIEW3D["3D geometry"]
    EXPLORER --> EVID["evidence inspection"]
    EXPLORER --> RUNCOMP["run comparison"]
```

## 11. Capacidades planejadas e relações futuras

As capacidades seguintes aparecem na arquitetura, mas ainda não possuem contract público completo equivalente aos módulos materializados.

```mermaid
flowchart LR
    CMAP["ConsolidatedContextMap"]

    CMAP -.-> MEMORY["semantic-memory\nplanned"]
    CMAP -.-> GRAPH["scene-graph\nplanned"]

    MEMORY -.-> REASON["context-reasoning\nplanned"]
    GRAPH -.-> REASON

    REASON -.-> QUERY["query-engine\nplanned"]

    QUERY -.-> Q1["semantic retrieval"]
    QUERY -.-> Q2["spatial questions"]
    QUERY -.-> Q3["navigation / hazard reasoning"]
```

Uma futura `scene-graph` deverá distinguir pelo menos:

```mermaid
flowchart TD
    R2D["relação observada em 2D"]
    R3D["relação confirmada geometricamente em 3D"]
    RCTX["relação inferida por contexto"]

    R2D --> REL["persistent relation + provenance"]
    R3D --> REL
    RCTX --> REL
```

Exemplos de relações de domínio pretendidas incluem `ON`, `ADJACENT_TO` e `BLOCKS`, sempre com proveniência e suporte, e não apenas uma aresta sem origem.

## 12. Dependências entre módulos em uma única visão

```mermaid
flowchart LR
    VP["visual-perception"]
    SE["state-estimation"]
    GM["geometric-map"]
    PR["point-representation"]
    SA["sensor-association"]
    SF["semantic-fusion"]
    SM["semantic-map"]

    SE -->|"pose + corrected LiDAR"| GM

    VP -->|"VisualObservation"| SA
    SE -->|"pose at RGB timestamp"| SA
    GM -->|"GeometryPoint + GeometryReference"| SA

    GM -->|"GeometryPoint"| PR
    PR -.->|"optional independent 3D evidence"| SF

    SA -->|"PointVisualAssociation / SemanticContribution"| SF
    GM -->|"3D neighborhood / geometry identity"| SF

    SF -->|"FusedPointContext / contextual run evidence"| SM
    GM -->|"shared geometry fingerprint"| SM
```

## 13. Ordem operacional de uma execução contextual real

```mermaid
sequenceDiagram
    participant Src as Bag/Dataset/Sensors
    participant RT as mapping-runtime
    participant SE as state-estimation
    participant GM as geometric-map
    participant VP as visual-perception
    participant SA as sensor-association
    participant SF as semantic-fusion
    participant SM as semantic-map
    participant Store as map-storage

    Src->>RT: RGB + LiDAR + IMU + timestamps
    RT->>RT: resolve window / keyframe identities

    RT->>SE: LiDAR + IMU
    SE-->>RT: trajectory + StateEstimate + corrected LiDAR

    RT->>GM: corrected geometry + pose context
    GM-->>RT: GeometryPoint[] + GeometryReference[]

    RT->>VP: RGB keyframes
    VP-->>RT: VisualObservation[]

    RT->>SA: GeometryPoint + pose + calibration + VisualObservation
    SA-->>RT: PointVisualAssociation[]

    RT->>SF: contributions grouped by GeometryReference
    SF-->>RT: FusedPointContext[]

    RT->>Store: publish contextual run

    Note over SM,Store: semantic-map operates across compatible published runs

    Store->>SM: PublishedContextRun A/B/...
    SM->>SM: verify map_id + map_frame + geometry_fingerprint
    SM-->>Store: ConsolidatedContextMap
```

## 14. O que não deve ser confundido

```mermaid
flowchart TD
    MASK["2D mask"] -. "não é" .-> XYZ["3D geometry"]
    DINO["DINO embedding"] -. "não é" .-> CLIP["CLIP embedding"]
    CLAIM["VLM confidence"] -. "não é automaticamente" .-> PROB["calibrated probability"]
    SIM["visual similarity"] -. "não implica" .-> ID["physical identity"]
    PVA["PointVisualAssociation"] -. "não é" .-> GT["ground truth"]
    FUSED["FusedPointContext"] -. "não apaga" .-> DISAGREE["source disagreement"]
    SMAP["semantic-map"] -. "não faz" .-> REG["cross-map registration"]
```

## Próxima leitura

- [`docs/README.md`](./README.md), índice estágio por estágio;
- [`modules/visual-perception/docs/pipeline-flow.md`](../modules/visual-perception/docs/pipeline-flow.md), fluxo interno completo de percepção visual;
- [`modules/visual-perception/docs/model-flow.md`](../modules/visual-perception/docs/model-flow.md), relação entre SAM3, DINOv2, CLIP e Qwen/Gemini;
- [`modules/state-estimation/docs/README.md`](../modules/state-estimation/docs/README.md);
- [`modules/geometric-map/docs/README.md`](../modules/geometric-map/docs/README.md);
- [`modules/sensor-association/docs/README.md`](../modules/sensor-association/docs/README.md);
- [`modules/semantic-fusion/docs/README.md`](../modules/semantic-fusion/docs/README.md);
- [`modules/semantic-map/docs/index.md`](../modules/semantic-map/docs/index.md);
- [`apps/mapping-runtime/README.md`](../apps/mapping-runtime/README.md).
