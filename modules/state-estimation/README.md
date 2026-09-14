# State Estimation

`state-estimation` fornece estimativas de movimento e pose necessárias para posicionar observações LiDAR em uma referência espacial consistente antes do mapeamento persistente e do processamento semântico.

> Documentação detalhada: [`docs/`](docs/README.md).

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

A associação câmera-LiDAR permanece de posse de `sensor-association`. A capacidade planejada de point representation permanece separada. A reconstrução geométrica persistente é de posse de `geometric-map`.

## Implementações externas

Sistemas concretos de odometria LiDAR-inercial são integrados por trás de adapters. O alvo inicial de integração é o FAST-LIO, mas os contracts públicos do módulo permanecem independentes de implementação, para que outros estimators, ground truth de simulador ou poses fornecidas por dataset possam ser substituídos mais tarde.

## Estrutura atual

```text
state-estimation/
├── README.md
├── configs/
├── docs/
├── src/
│   └── state_estimation/
│       ├── application/
│       ├── domain/
│       ├── ports.py
│       ├── ports/
│       ├── infrastructure/
│       │   └── fast_lio/
│       └── cli/
├── tests/
└── benchmarks/
```

Nem toda subpasta representa uma camada com código ativo. A documentação local distingue o que está implementado de estruturas ainda usadas apenas para organizar responsabilidades futuras.