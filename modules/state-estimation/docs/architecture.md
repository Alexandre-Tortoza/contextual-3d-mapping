# State Estimation Architecture

`state-estimation` is the geometric motion boundary between raw LiDAR/IMU observations and downstream 3D semantic processing.

## Internal flow

```mermaid
flowchart LR
    L[LiDAR observation] --> A[Input adapters]
    I[IMU observation] --> A

    A --> V[Validation and time checks]
    V --> P[StateEstimator port]

    P --> B[LiDAR-inertial backend adapter]
    B --> S[State estimate]
    B --> C[Motion-corrected LiDAR frame]

    S --> O[Public outputs]
    C --> O
```

The backend adapter is replaceable. FAST-LIO is the initial concrete integration target, but no downstream consumer should depend on FAST-LIO-specific messages, configuration structures, ROS types, or internal state.

## Public outputs

The module is expected to expose contract-compatible information conceptually equivalent to:

```text
StateEstimate
├── timestamp
├── pose
├── velocity?
├── covariance?
├── coordinate_frame
├── validity
└── provenance

MotionCorrectedLiDARFrame
├── timestamp
├── points
├── coordinate_frame
├── pose_reference
└── provenance
```

Exact schemas are defined by implementation issues rather than this document.

## Boundary with point-representation

`point-representation` consumes point geometry and produces learned point embeddings. It must not own odometry, pose estimation, IMU fusion, scan deskewing, or trajectory estimation.

```mermaid
flowchart LR
    SE[state-estimation] -->|motion-corrected LiDAR| PR[point-representation]
    PR -->|point embeddings| SA[sensor-association]
    SE -->|pose / trajectory| SA
```

## Boundary with sensor-association

`state-estimation` may provide the platform trajectory and LiDAR-side pose information required for multimodal alignment. It does not calibrate the camera to the LiDAR and does not generate point-to-pixel or point-to-feature correspondences.

```mermaid
flowchart LR
    SE[state-estimation] -->|trajectory / LiDAR pose| SA[sensor-association]
    VP[visual-perception] -->|visual observations| SA
    PR[point-representation] -->|point representations| SA
    CAL[camera-LiDAR calibration] --> SA
```

## Boundary with geometric mapping

The module may emit motion-corrected or pose-associated LiDAR frames, but it does not own a persistent global geometric map. If persistent geometric reconstruction becomes a first-class capability, it should be introduced as a separate module and consume state-estimation outputs through contracts.

## External backend isolation

The initial backend integration must be isolated under infrastructure code. The adapter is responsible for translating external runtime inputs and outputs into module contracts.

Backend-specific concerns include:

- middleware message types;
- process lifecycle;
- sensor topic names;
- LiDAR-IMU extrinsics;
- backend configuration files;
- runtime diagnostics;
- external dependency installation.

None of these concerns should leak into the domain or downstream modules.
