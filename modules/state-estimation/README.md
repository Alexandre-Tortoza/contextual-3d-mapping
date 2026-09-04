# State Estimation

`state-estimation` fornece estimativas de movimento e pose necessárias para posicionar observações LiDAR em uma referência espacial consistente antes do mapeamento persistente e do processamento semântico.

O módulo é independentemente executável, testável, avaliável (benchmarkable) e substituível. Módulos downstream dependem de seus contracts públicos, não de uma implementação específica de odometria.

## Responsabilidades

- consumir observações LiDAR e IMU com timestamp;
- estimar a pose e a trajectory do sensor/plataforma;
- expor observações LiDAR corrigidas por movimento quando suportado pelo backend selecionado;
- preservar metadados de frame de coordenadas, timestamp, incerteza e proveniência;
- expor informação de saúde e validade para consumidores downstream.

## Não-responsabilidades

- calibração câmera-LiDAR ou geração de correspondência visual;
- point embeddings aprendidos;
- fusão semântica;
- construção de mapa geométrico ou semântico persistente;
- scene graphs, memória semântica ou raciocínio contextual.

A associação câmera-LiDAR permanece de posse de `sensor-association`. Features LiDAR aprendidas permanecem de posse de `point-representation`. A reconstrução geométrica persistente é de posse de `geometric-map`.

## Implementações externas

Sistemas concretos de odometria LiDAR-inercial são integrados por trás de adapters. O alvo inicial de integração é o FAST-LIO, mas os contracts públicos do módulo permanecem independentes de implementação, para que outros estimators, ground truth de simulador, ou poses fornecidas por dataset possam ser substituídos mais tarde.

## Estrutura inicial

```text
state-estimation/
├── README.md
├── configs/
├── docs/
├── src/
│   └── state_estimation/
│       ├── application/
│       ├── domain/
│       ├── ports/
│       ├── infrastructure/
│       │   └── fast_lio/
│       └── cli/
├── tests/
└── benchmarks/
```

O manifest do pacote e os arquivos de implementação concreta devem ser introduzidos apenas quando as issues de implementação correspondentes forem tratadas.
