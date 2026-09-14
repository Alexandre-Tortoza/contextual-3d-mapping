# Geometric Map

`geometric-map` possui a geometria persistente do mundo, montada a partir de estimativas de movimento e observações LiDAR compatíveis com o contract.

> Documentação detalhada: [`docs/`](docs/README.md).

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

`state-estimation` fornece o contexto de movimento. A capacidade futura de `semantic-map` deverá enriquecer a geometria através de referências estáveis, em vez de duplicar o ownership geométrico.

## Implementação atual

O primeiro slice usa `InMemoryGeometricMap` e contracts explícitos em `models.py`.

```text
geometric-map/
├── README.md
├── docs/
│   └── README.md
├── src/
│   └── geometric_map/
│       ├── __init__.py
│       ├── models.py
│       └── in_memory.py
├── tests/
│   ├── README.md
│   └── test_in_memory.py
├── benchmarks/
│   └── README.md
└── pyproject.toml
```

Diretórios vazios reservados para `application`, `domain`, `ports`, `infrastructure` e `configs` foram removidos. Eles só devem voltar quando houver uma responsabilidade e conteúdo concretos.

Estruturas de índice, estratégias de reconstrução, formatos de persistência e novos backends podem evoluir sem alterar os contracts públicos consumidos pelos demais módulos.