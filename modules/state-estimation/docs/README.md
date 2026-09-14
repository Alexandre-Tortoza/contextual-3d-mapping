# State Estimation

Esta documentação descreve a fronteira e a implementação atual de `state-estimation`. A posição do módulo na pipeline completa está em [`docs/16-pose-calibration.md`](../../../docs/16-pose-calibration.md).

## Papel no sistema

`state-estimation` transforma observações LiDAR e IMU com timestamp em contexto de movimento utilizável pelos módulos geométricos.

```text
LiDAR + IMU
    -> state-estimation
        -> pose
        -> trajectory
        -> MotionCorrectedLidarFrame
    -> geometric-map
```

O módulo não é dono da calibração câmera-LiDAR nem da associação visual. Essas responsabilidades ficam em `sensor-association`.

## Fronteira pública

A API pública está concentrada em:

- [`models.py`](../src/state_estimation/models.py), value objects e resultados do estimator;
- [`ports.py`](../src/state_estimation/ports.py), ponto de substituição de implementações de odometria;
- [`__init__.py`](../src/state_estimation/__init__.py), exports estáveis consumidos pelos outros módulos.

Os contracts preservam frame, timestamp, pose, proveniência e estado de validade para que consumidores downstream não dependam de detalhes do backend.

## Backend FAST-LIO

A integração concreta atual fica em [`infrastructure/fast_lio/`](../src/state_estimation/infrastructure/fast_lio/).

`adapter.py` isola o formato específico do FAST-LIO da API pública do módulo. Isso permite substituir o estimator por outra implementação, ground truth de simulador ou poses fornecidas por dataset sem alterar `geometric-map` e os demais consumidores.

## Configuração

A configuração versionada do primeiro slice está em [`configs/`](../configs/), incluindo o perfil `corridor-02-velodyne.yaml`.

Configuração de sensor e backend deve permanecer local ao módulo quando afeta a estimação de estado. Calibração usada especificamente para projeção câmera-LiDAR pertence à fronteira de `sensor-association`.

## Estrutura atual

```text
modules/state-estimation/
├── README.md
├── docs/
│   └── README.md
├── configs/
├── src/
│   └── state_estimation/
│       ├── models.py
│       ├── ports.py
│       └── infrastructure/
│           └── fast_lio/
├── tests/
└── benchmarks/
```

Alguns diretórios ainda contêm apenas notas de responsabilidade para evolução futura. Eles não devem ser tratados como camadas implementadas. Quando uma separação interna deixar de representar código real, a estrutura deve ser simplificada em vez de ser preservada apenas por antecipação.

## Invariantes de integração

- timestamps e `clock_id` precisam permanecer explícitos;
- transformações devem declarar frame de origem e destino;
- downstream consome contracts públicos, não objetos internos do FAST-LIO;
- correção de movimento deve ser indicada no resultado, não presumida;
- incerteza, validade e proveniência não devem ser descartadas na adaptação.

## Relação com a documentação principal

- [`docs/16-pose-calibration.md`](../../../docs/16-pose-calibration.md), pose e calibração na pipeline;
- [`docs/15-geometric-map.md`](../../../docs/15-geometric-map.md), consumidor imediato do contexto de movimento;
- [`docs/17-sensor-association.md`](../../../docs/17-sensor-association.md), uso de pose para alinhar mapa e câmera.

Decisões internas do estimator ficam neste diretório. A narrativa que atravessa sensores e módulos permanece em [`docs/`](../../../docs/README.md).