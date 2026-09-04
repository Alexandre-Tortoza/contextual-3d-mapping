# Guia de integração

Este guia mostra o fluxo de consumo de `visual-perception`. Ele não cria um CLI, não
resolve URIs de imagem e não seleciona um dataset: essas decisões são de posse da
aplicação ou de um adapter.

## Caminho mínimo com fakes

Instale o módulo no diretório `modules/visual-perception`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

A aplicação resolve os pixels, cria os contracts compartilhados de origem e então chama
a API pública. O exemplo abaixo usa `fake`, o default determinístico e livre de GPU.

```python
import numpy as np
from contextual_mapping_contracts import (
    FrameId,
    ObservationReference,
    SourceArtifactReference,
    Timestamp,
)
from visual_perception import (
    ImageObservation,
    ImagePayload,
    ModuleConfig,
    create_perception_ports,
    run_canonical_pipeline,
)

pixels = np.zeros((480, 640, 3), dtype=np.uint8)
source = ObservationReference(
    observation_id="camera-001",
    dataset_id="example",
    sequence_id="sequence-001",
    sensor_id="front-camera",
    sequence_index=0,
    timestamp=Timestamp(nanoseconds=0, clock_id="system"),
    frame_id=FrameId("front_camera_optical_frame"),
)
image = ImageObservation(
    width=640,
    height=480,
    encoding="rgb8",
    image=SourceArtifactReference(uri="memory://camera-001", media_type="image/raw"),
    source=source,
)
payload = ImagePayload(pixels=pixels, width=640, height=480)
config = ModuleConfig()
result = run_canonical_pipeline(image, payload, config, create_perception_ports(config))

if not result.audit.passed:
    raise RuntimeError(result.audit.errors)

observation = result.observation
for warning in result.audit.warnings:
    print(warning.code, warning.message)
```

O exemplo usa apenas exports do pacote e contracts de nível de repositório. Em uma
aplicação real, substitua os valores sintéticos por metadados preservados pelo adapter;
não invente timestamp, frame ou URI após o recebimento da imagem.

## Entrada vinda de adapters

Quando a entrada já é um `contextual_mapping_adapters.CanonicalObservation` de kind
`"rgb"`, a fronteira
[`to_canonical_input`](../src/visual_perception/infrastructure/integration/rgb_adapter_boundary.py)
constrói o par `ImageObservation`/`ImagePayload`. A aplicação ainda resolve
`observation.artifact` para um array RGB antes da chamada. Um kind diferente de `"rgb"`,
pixels fora de `(H, W, 3)` ou dimensões incompatíveis falham antes da inferência.

Essa helper pertence à integração porque seu tipo de entrada é de outro módulo; o
pipeline depende apenas dos contracts públicos de `visual-perception`.

## Backends reais

Backends reais são opt-in. Instale as dependências necessárias e forneça checkpoint e
backend explícitos:

```bash
pip install -e ".[ml,sam2,openclip]"
```

Os identificadores aceitos pela factory são `sam2`, `dinov2`, `openclip` e `qwen_vl`.
Cada adapter carrega lazy e exige um `checkpoint` diferente de `"none"`; `device="auto"`
só escolhe GPU quando ela está disponível. Ausência de dependência, GPU solicitada
indisponível ou checkpoint ausente resulta em `BackendUnavailableError`; falha de
carregamento ou inferência resulta em `BackendExecutionError`. Não há fallback
silencioso para fake.

Os nomes, dependências específicas e limites atuais estão em
[model-backends.md](model-backends.md). A seleção de checkpoint de referência ainda
depende de benchmark, portanto registre `ModuleConfig` e seu fingerprint junto com o
artifact gerado.

## Consumidores downstream e persistência

`VisualObservation` entrega masks, boxes, timestamp, frame e convenção de coordenadas a
`sensor-association`; o fixture de fronteira em
[`sensor_association_contract.py`](../src/visual_perception/infrastructure/integration/sensor_association_contract.py)
mostra o conjunto mínimo de dados transferidos. Relações continuam candidatas 2D até a
verificação downstream.

Para persistir, use a integração em
[`persistence_integration.py`](../src/visual_perception/infrastructure/integration/persistence_integration.py)
com uma implementação de `EvidencePersistencePort`. A observação serializada preserva
geometria e metadata; vetores de embedding são persistidos separadamente e continuam
referenciados por id. Veja [artifacts.md](artifacts.md).

Um composition root pode converter `VisualPerceptionError` em seu próprio resultado de
falha. A referência de como `mapping-runtime` faz isso sem expor exceptions de backend
está em
[`mapping_runtime_integration.py`](../src/visual_perception/infrastructure/integration/mapping_runtime_integration.py).
