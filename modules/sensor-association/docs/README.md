# Sensor Association

Esta documentação descreve a implementação local de `sensor-association`. O fluxo end-to-end está em [`docs/17-sensor-association.md`](../../../docs/17-sensor-association.md) e [`docs/18-point-visual-association.md`](../../../docs/18-point-visual-association.md).

## Papel no sistema

O módulo conecta geometria 3D e evidência visual sem decidir a semântica final do mapa.

```text
GeometryPoint + pose + calibração + VisualObservation
    -> sensor-association
        -> projeção
        -> visibilidade
        -> oclusão
        -> associação de região/cor
    -> PointVisualAssociation
    -> semantic-fusion
```

## Implementação atual

A implementação está concentrada em dois arquivos:

- [`models.py`](../src/sensor_association/models.py), contracts de câmera, calibração, associação e motivos de rejeição;
- [`projector.py`](../src/sensor_association/projector.py), projeção e regras de visibilidade/oclusão.

A API diferencia dois casos:

- `associate_points`, para um scan LiDAR recém-chegado;
- `associate_map_points`, para pontos já persistidos no mapa geométrico.

Essa separação é intencional. Um mapa acumulado é esparso e precisa de uma regra de oclusão mais conservadora que um scan observado de um único viewpoint.

## Modelos de câmera

O módulo suporta projeção calibrada para os modelos expostos pelo contract, incluindo pinhole, fisheye equidistante e MEI. A associação só é válida quando os frames e a proveniência da calibração são compatíveis com os dados recebidos.

## Regras de visibilidade

Antes de anexar evidência visual a um ponto, a implementação verifica:

- hemisfério frontal da câmera;
- domínio válido do modelo óptico;
- limites da imagem;
- consistência temporal entre LiDAR e RGB;
- oclusão;
- suporte visual disponível no pixel resultante.

Rejeições são resultados explícitos, com motivo, e não associações parciais. Um ponto rejeitado não deve carregar evidência visual residual.

## Oclusão

Para scans, o z-buffer por pixel é suficiente para a fronteira atual. Para pontos de mapa, `associate_map_points` agrega profundidade em células e consulta uma vizinhança para reduzir vazamento através de superfícies esparsas.

A profundidade comparada é a coordenada no eixo óptico. Os parâmetros de célula e tolerância são parte da configuração porque dependem da densidade geométrica.

## Estrutura atual

```text
modules/sensor-association/
├── README.md
├── docs/
│   └── README.md
├── src/
│   └── sensor_association/
│       ├── __init__.py
│       ├── models.py
│       └── projector.py
└── tests/
    └── test_projector.py
```

## Testes e contracts

`tests/test_projector.py` cobre projeção, filtros, oclusão e equivalência entre as duas fronteiras de associação. Alterações no modelo de câmera ou na política de oclusão devem manter esses invariantes ou atualizar explicitamente o contract.

## Relação com a documentação principal

- [`docs/16-pose-calibration.md`](../../../docs/16-pose-calibration.md), transformação mapa-câmera e calibração;
- [`docs/17-sensor-association.md`](../../../docs/17-sensor-association.md), transformação desta etapa;
- [`docs/18-point-visual-association.md`](../../../docs/18-point-visual-association.md), payload entregue downstream;
- [`docs/19-semantic-fusion.md`](../../../docs/19-semantic-fusion.md), consumidor da associação.

Detalhes de projeção e políticas do módulo pertencem a este diretório. A narrativa entre módulos permanece em [`docs/`](../../../docs/README.md).

## Próxima leitura

- [17. Sensor Association](../../../docs/17-sensor-association.md)
- [18. PointVisualAssociation](../../../docs/18-point-visual-association.md)
- [Documentação de `semantic-fusion`](../../semantic-fusion/docs/README.md)