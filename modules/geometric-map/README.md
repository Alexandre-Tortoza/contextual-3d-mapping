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

## Implementação atual

```text
src/geometric_map/
├── __init__.py      API pública
├── models.py        pontos, referências e bounds
├── in_memory.py     mapa persistente em memória
└── point_cloud.py   leitura integral de PCD e proveniência
```

A leitura de arquivos e o mapa em memória compartilham a identidade geométrica
pública, com suas estruturas de armazenamento locais ao módulo.

## Leitura de geometria medida

`read_pcd_geometry(path, map_id=..., frame_id=..., expected_sha256=...)` lê um
PCD `DATA binary` completo. O parser, seus offsets e a validação do payload
pertencem a este módulo. Campos XYZ devem ser escalares; campos vetoriais
adicionais são considerados no tamanho do registro.

`PointCloudGeometry` publica XYZ finito em metros, `source_indices` originais,
contagem original, frame, mapa, referência e digest SHA-256. Arrays são
copiados e protegidos contra escrita. A remoção de NaNs não renumera registros;
`source_indices` permite preservar a identidade `map_id:pcd:índice` no exportador.
Hash divergente ou payload truncado falha antes da associação.

O exportador do runtime escolhe a amostra de renderização desta mesma leitura.
`sensor-association` recebe a geometria completa: o número de pontos do viewer
não determina a densidade dos oclusores.
