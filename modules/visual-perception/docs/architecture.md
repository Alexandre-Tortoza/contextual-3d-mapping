# Arquitetura

## Responsabilidade e fronteira

`visual-perception` recebe uma imagem RGB já identificada, datada e associada ao frame
do sensor. Ele produz uma `VisualObservation` com regiões 2D, embeddings referenciados,
claims semânticos de cena e região, relações candidatas e o resultado de uma auditoria.
O ponto de entrada de produção é
[`run_canonical_pipeline`](../src/visual_perception/application/pipeline.py).

O módulo não lê datasets, ROS bags ou URIs de artifact; essa é a responsabilidade de
[`adapters`](../../../adapters/README.md). Também não calibra sensores, associa pixels a
geometria 3D, persiste o mapa, verifica relações em 3D ou constrói scene graphs. Essas
capacidades pertencem, respectivamente, a `sensor-association`, módulos de mapa e
`scene-graph`/`context-reasoning`.

```text
CanonicalObservation RGB + pixels resolvidos
    -> ImageObservation + ImagePayload
    -> visual-perception
    -> VisualObservation + AuditResult
    -> sensor-association / persistência / mapping-runtime
```

Veja [integration.md](integration.md) para as fronteiras de entrada e saída e
[api-contracts.md](api-contracts.md) para os tipos do fluxo.

## Organização interna

```text
src/visual_perception/
├── domain/           # contracts, tipos e invariantes do domínio visual
├── ports/            # Protocols para backends substituíveis
├── application/      # estágios e orquestração do pipeline
├── infrastructure/
│   ├── fakes/        # implementações determinísticas, sem GPU
│   ├── adapters/     # runtimes reais isolados de bibliotecas externas
│   ├── integration/  # fronteiras com adapters, runtime e persistência
│   └── serialization.py
└── config.py          # configuração validada e reproduzível do módulo
```

`domain/` é dono de geometria de imagem, regiões, embeddings, claims, relações e
proveniência de modelo. `ports/` define os quatro pontos de variação reais: descoberta
de regiões, feature extraction densa, encoding alinhado à linguagem e raciocínio
multimodal. `application/` depende desses contracts, não dos runtimes de terceiros. A
factory [`create_perception_ports`](../src/visual_perception/infrastructure/adapters/factory.py)
é o local de composição entre configuração e implementações concretas.

Os identificadores, timestamp, frame e referências de artifacts são contracts
compartilhados, definidos em [`contracts/`](../../../contracts/README.md), pois têm o
mesmo significado para mais de um módulo. Não introduza cópias locais desses conceitos.

## Invariantes de coordenadas

Toda máscara, box e transform obedece à convenção `top-left-origin,half-open-xyxy`,
registrada em cada `VisualObservation`:

- `(0, 0)` é o pixel superior esquerdo; `x` cresce à direita e `y` para baixo;
- `BoundingBox` usa `(x_min, y_min, x_max, y_max)` com mínimo inclusivo e máximo
  exclusivo, compatível com `array[y_min:y_max, x_min:x_max]`;
- `Mask` é booleana, tem shape `(height, width)` e usa a resolução integral da imagem;
- transformações entre tile e imagem global passam por `CoordinateTransform`; não se
  deve aplicar offsets ou escalas ad hoc em consumidores downstream.

Esses invariantes são validados por
[`domain/geometry.py`](../src/visual_perception/domain/geometry.py) e pela construção de
[`VisualObservation`](../src/visual_perception/domain/visual_observation.py). Uma
relação só pode referenciar regiões presentes na mesma observação.

## Decisões de representação

Uma região preserva `geometric_confidence`, que mede a confiança da máscara e box,
independentemente da confiança de cada `SemanticClaim`. Claims não são reduzidos a um
único label: hipóteses conflitantes continuam visíveis para a auditoria. Relações são
candidatas no plano da imagem; mesmo uma relação de fonte `geometric_2d` não é uma
relação 3D confirmada.

Embeddings visuais e de linguagem ficam fora do payload canônico. A observação armazena
somente referências estáveis para que o vetor possa ter ciclo de vida e armazenamento
adequados ao consumidor. Consulte [artifacts.md](artifacts.md) para a persistência e
[research-traceability.md](research-traceability.md) para a motivação dessas escolhas.
