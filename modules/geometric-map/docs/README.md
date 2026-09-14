# Geometric Map

Esta documentação descreve a implementação local de `geometric-map`. A visão end-to-end permanece em [`docs/15-geometric-map.md`](../../../docs/15-geometric-map.md).

## Papel no sistema

`geometric-map` é o dono da geometria persistente. Ele recebe observações LiDAR já corrigidas por movimento por `state-estimation`, aplica a pose ao frame do mapa e expõe referências estáveis para consumidores como `sensor-association`.

```text
state-estimation
    -> MotionCorrectedLidarFrame
    -> geometric-map
        -> GeometryPoint
        -> GeometryReference
        -> Bounds3D
    -> sensor-association
```

## Implementação atual

O primeiro slice é deliberadamente simples e determinístico:

- [`models.py`](../src/geometric_map/models.py) define os contracts públicos `GeometryReference`, `GeometryPoint` e `Bounds3D`;
- [`in_memory.py`](../src/geometric_map/in_memory.py) implementa `InMemoryGeometricMap`;
- `insert()` transforma cada ponto do frame LiDAR para o frame global usando a pose publicada pelo estimator;
- `lookup()` recupera pontos por bounds 3D inclusivos;
- `bounds()` expõe a extensão espacial atual sem vazar o armazenamento interno;
- a identidade de cada ponto persiste como `<observation_id>:<point_index>` dentro de um `MapId`.

A implementação em memória é um primeiro backend. Estruturas de índice espacial, reconstrução, persistência em disco ou outros backends devem ser adicionados quando houver uma necessidade concreta, preservando a API pública.

## Invariantes importantes

- coordenadas são expressas em metros;
- pontos persistidos pertencem ao `FrameId` do mapa;
- a pose recebida precisa transformar do frame do LiDAR corrigido para o frame do mapa;
- cada `GeometryPoint` preserva a coordenada de origem, a observação LiDAR e a proveniência;
- `GeometryReference` identifica geometria sem expor a estrutura interna do mapa;
- um mapa vazio retorna `None` para bounds, em vez de inventar uma extensão espacial.

## Estrutura real

```text
modules/geometric-map/
├── README.md
├── docs/
│   └── README.md
├── src/
│   └── geometric_map/
│       ├── __init__.py
│       ├── models.py
│       └── in_memory.py
├── tests/
│   └── test_in_memory.py
└── benchmarks/
    └── README.md
```

Diretórios reservados sem implementação foram removidos. Novas camadas devem existir apenas quando houver código ou documentação concreta que justifique sua responsabilidade.

## Testes

Os testes de `tests/test_in_memory.py` são a referência executável para inserção, transformação, identidade estável e lookup espacial.

## Relação com a documentação principal

- [`docs/15-geometric-map.md`](../../../docs/15-geometric-map.md), posição do módulo na pipeline completa;
- [`docs/16-pose-calibration.md`](../../../docs/16-pose-calibration.md), origem do contexto de pose usado na transformação;
- [`docs/17-sensor-association.md`](../../../docs/17-sensor-association.md), consumidor da geometria persistente.

A documentação deste diretório deve explicar decisões internas de `geometric-map`. Fluxos que atravessam múltiplos módulos permanecem em [`docs/`](../../../docs/README.md).