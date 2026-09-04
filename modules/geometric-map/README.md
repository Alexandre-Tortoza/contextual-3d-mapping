# Geometric Map

`geometric-map` possui a geometria persistente do mundo, montada a partir de estimativas de movimento e observações LiDAR compatíveis com o contract.

## Responsabilidades

- consumir pose, trajectory e observações LiDAR através de contracts públicos;
- posicionar observações em um frame de mundo consistente;
- manter o estado persistente do mapa geométrico;
- expor identificadores de geometria estáveis e bounds espaciais;
- preservar a proveniência da geometria e referências à observação de origem;
- fornecer geometria adequada para associação semântica, mapeamento, avaliação e visualização downstream.

## Não-responsabilidades

- estimar movimento a partir de entradas brutas de LiDAR/IMU;
- representação de pontos aprendida;
- percepção visual;
- classificação ou fusão semântica;
- memória semântica, scene graphs ou raciocínio contextual;
- renderização voltada ao usuário.

`state-estimation` fornece o contexto de movimento. `semantic-map` enriquece a geometria através de referências estáveis, em vez de duplicar o ownership geométrico.

## Estrutura inicial

```text
geometric-map/
├── README.md
├── configs/
├── docs/
├── src/
│   └── geometric_map/
│       ├── application/
│       ├── domain/
│       ├── ports/
│       └── infrastructure/
├── tests/
└── benchmarks/
```

Estruturas de dados concretas, índices, estratégias de reconstrução, formatos de persistência e implementações devem ser introduzidos por issues de implementação, mantendo os contracts públicos independentes de implementação.
