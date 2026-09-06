# Execução

Este documento explica **como o módulo é executado, como os modelos são carregados,
como cache e fingerprints funcionam e onde diagnosticar falhas de runtime**.

Para a ordem semântica dos estágios, consulte [pipelines.md](pipelines.md). Para seleção
de checkpoints, consulte [model-backends.md](model-backends.md).

## Modos de execução

Existem dois cenários principais:

```text
fake
    -> desenvolvimento, testes, integração e contracts
    -> sem GPU
    -> determinístico

real
    -> inferência dos backends selecionados
    -> GPU quando configurada
    -> lifecycle, métricas e artifacts de validação
```

`ModuleConfig()` usa fakes por padrão.

A configuração real de referência pode ser criada com:

```python
from visual_perception.application.execution_profile import research_quality_config

config = research_quality_config(real_backends=True)
```

A composição dos ports ocorre antes da chamada do pipeline:

```python
ports = create_perception_ports(config)
result = run_canonical_pipeline(image, payload, config, ports)
```

`run_canonical_pipeline` recebe backends já compostos. Ele não escolhe checkpoints nem
instancia adapters concretos por nome.

## Fluxo de execução

```mermaid
flowchart TD
    C[ModuleConfig] --> P[create_perception_ports]
    P --> L[ModelLifecycleManager compartilhado]
    P --> R[PerceptionPorts]
    R --> X[run_canonical_pipeline]
    X --> S1[region discovery]
    S1 --> S2[dense features]
    S2 --> S3[language embedding]
    S3 --> S4[multimodal reasoning]
    X --> O[PipelineResult]
```

A ordem exata e as dependências entre todos os estágios estão em
[pipelines.md](pipelines.md#fluxo-canônico).

## Ciclo de vida dos modelos

[`application/lifecycle.py`](../src/visual_perception/application/lifecycle.py) define
`ModelLifecycleManager`.

Seu papel é:

- carregar um backend pesado somente quando necessário;
- registrar métricas do estágio;
- liberar a referência ao final do uso;
- evitar manter todos os modelos pesados residentes ao mesmo tempo.

A factory
[`create_perception_ports`](../src/visual_perception/infrastructure/adapters/factory.py)
recebe um manager opcional. Quando omitido, cria um manager e o compartilha entre os
ports compostos.

Essa distinção é importante:

```text
create_perception_ports(...)
    -> compõe adapters
    -> fornece/compartilha lifecycle manager

run_canonical_pipeline(...)
    -> recebe PerceptionPorts prontos
    -> não escolhe backends
```

## Execução sequencial e VRAM

Na configuração real de referência, os quatro modelos cabem individualmente em 8GB, mas
a soma dos picos excede esse orçamento. O comportamento esperado é conceitualmente:

```text
SAM
  -> load
  -> inferência
  -> métricas
  -> unload

DINOv2
  -> load
  -> inferência
  -> métricas
  -> unload

CLIP
  -> load
  -> inferência
  -> métricas
  -> unload

Qwen-VL
  -> load
  -> inferência
  -> métricas
  -> unload
```

Os valores medidos e a motivação da seleção estão em
[model-backends.md](model-backends.md#orçamento-de-vram-e-lifecycle).

## Métricas de estágio

`ModelLifecycleManager.metrics` registra informações úteis para auditoria de runtime.
Dependendo do ambiente e adapter, isso inclui:

- nome do estágio;
- backend/checkpoint;
- tempo de carregamento;
- tempo de inferência;
- pico de memória observado.

Em ambientes sem CUDA, memória de processo pode ser apenas uma proxy. Resultados de VRAM
usados para decisões de referência devem vir da validação real em GPU e registrar o
hardware utilizado.

## Cache de estágio

[`application/cache.py`](../src/visual_perception/application/cache.py) define
`StageCache`.

O cache grava um registro por estágio concluído, indexado por fingerprint.

O fingerprint de um estágio inclui:

- versão/configuração própria;
- fingerprints das dependências upstream relevantes.

Consequências:

```text
execução idêntica
    -> reutiliza cache válido

mudança em um estágio
    -> invalida o estágio
    -> invalida dependentes downstream
    -> não precisa invalidar irmãos independentes

execução interrompida
    -> próxima execução pode reutilizar estágios válidos
```

`CACHE_SCHEMA_VERSION` impede reutilização silenciosa de uma entrada escrita por uma
versão incompatível do cache.

## Cache não é persistência canônica

Não confunda:

```text
StageCache
    = otimização de execução
    = artefato intermediário por estágio

VisualObservation persistida
    = produto canônico serializável do módulo
    = contract entre processos/consumidores
```

Para persistência canônica, consulte [artifacts.md](artifacts.md).

## Fingerprints e reprodutibilidade

Uma mudança que altera resultado deve participar da cadeia de fingerprint apropriada.
Exemplos:

- checkpoint;
- threshold;
- versão de prompt;
- configuração de tiling;
- parâmetro de merge;
- implementação/versionamento de estágio quando aplicável.

O objetivo é evitar este erro:

```text
configuração mudou
    + fingerprint permaneceu igual
    -> cache antigo reutilizado como se fosse resultado novo
```

Artifacts de benchmark ou validação devem registrar configuração e revisão do código,
não apenas a saída visual.

## Falhas de runtime

A hierarquia de erros está em
[`domain/errors.py`](../src/visual_perception/domain/errors.py).

Fluxo esperado:

```text
dependência/checkpoint/device indisponível
    -> BackendUnavailableError

falha durante load ou inferência
    -> BackendExecutionError

resposta semântica malformada de uma região
    -> RegionInterpretationFailure
    -> restante da observação pode continuar
```

Não substitua silenciosamente backend real por fake quando ocorre uma falha.

## Auditoria da saída

Depois da execução, `PipelineResult.audit` contém `AuditResult`.

```python
result = run_canonical_pipeline(...)

if not result.audit.passed:
    handle_invalid_observation(result.audit.errors)

for warning in result.audit.warnings:
    record_warning(warning)
```

Warnings fazem parte da rastreabilidade. Um consumidor não deve descartá-los apenas
porque `passed=True`.

## Validação local sem GPU

No diretório do módulo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

Os fakes permitem validar o pipeline e os contracts sem baixar modelos.

## Execução com backends reais

Instale as dependências de ML:

```bash
pip install -e ".[ml]"
```

Use a configuração de referência ou uma `ModuleConfig` explícita. Não esconda
checkpoint ou device em estado global.

Antes de tratar um resultado como comparável a outro, registre:

- git revision;
- `ModuleConfig` e fingerprint;
- checkpoints;
- device/hardware;
- dataset/subset;
- métricas de lifecycle;
- versão dos artifacts de saída.

## Benchmark de backends

Para comparar implementações concretas:

```text
benchmarks/run_backend_benchmark.py
benchmarks/backend_benchmark.py
benchmarks/candidates/
```

Resultados que orientam configuração de referência devem ser versionados e conter
contexto experimental suficiente para reprodução.

A seleção atual está documentada em [model-backends.md](model-backends.md).

## Validação end-to-end

A pipeline real de referência pode ser exercitada nos três frames de auditoria por:

```bash
python benchmarks/validate_reference_pipeline.py \
  --frame-id corridor-02-000 \
  --frame-id corridor-02-008 \
  --frame-id corridor-02-017
```

A execução gera samples em:

```text
benchmarks/results/samples/<run-id>/
```

`manifest.json` preserva IDs na ordem solicitada, SHA-256 de cada entrada, revisão Git,
configuração e fingerprint, latência, VRAM, falhas, audit e estados dos slots por frame.
As observações e overlays emitidos diretamente pelo pipeline ficam em `canonical/`.
O merge semântico externo só roda com `--semantic-merge` e fica separado em
`postprocessed/semantic-merge/`; portanto ele nunca substitui silenciosamente o artifact
canônico. Esses artifacts são úteis para inspeção qualitativa e diagnóstico, mas não
substituem protocolos de avaliação quantitativa específicos.

Para a ablation operacional da #209, execute os mesmos IDs com
`--context-profile baseline` e `--context-profile full`. O manifest registra o perfil,
os estados, chamadas e latência de cada slot; IDs e geometria continuam sendo a saída
canônica anterior à evidência contextual.

O resultado medido na RTX 3060 está em
[`../benchmarks/results/benchmark-209-multi-context-20260906T201128Z.md`](../benchmarks/results/benchmark-209-multi-context-20260906T201128Z.md).
Nos três frames, o perfil full produziu os quatro slots em 126/126 regiões, sem falhas,
preservou exatamente a geometria canônica e adicionou 5,489 s (+1,09%) ao wall-clock,
sem aumentar o pico observado de 4,57 GiB.

## Quero diagnosticar X: onde olhar?

| Sintoma | Primeiro local | Depois verificar |
| --- | --- | --- |
| OOM ao trocar de estágio | `application/lifecycle.py` | factory, `_runtime.py`, métricas e configuração |
| cache não invalida após mudança | `application/cache.py` | fingerprint do estágio e `ModuleConfig` |
| resultado antigo reaparece | `StageCache` | versão de cache e dependências upstream |
| backend real virou fake | `factory.py` | isso não deve ocorrer silenciosamente |
| erro ao carregar checkpoint | adapter concreto | `BackendUnavailableError` / `BackendExecutionError` |
| uma região falha mas pipeline continua | `application/region_semantics.py` | `region_interpretation_failures` |
| auditoria reprova resultado | `application/quality_audit.py` | `AuditResult.errors` |
| qualidade caiu após trocar modelo | benchmark da capability | configuração, dataset e proxy usado |
| VRAM aumentou | lifecycle metrics | checkpoint, dtype, quantização e sequência de load |

## Checklist de execução reproduzível

- use `ModuleConfig` explícita para experimentos;
- registre fingerprint;
- preserve git revision;
- registre checkpoint e versão de prompt;
- registre hardware e device;
- não reutilize cache incompatível;
- preserve `AuditResult` e warnings;
- mantenha artifacts de benchmark ligados ao dataset/subset usado;
- compare resultados somente sob protocolo equivalente.

## Ajuste e validação da calibração semântica (#211)

O ajuste aceita somente um manifest integralmente `reviewed` e um arquivo de predições
que contenha exatamente as amostras do split `calibration`. Amostras de `development` ou
`test` no arquivo de ajuste causam erro antes da construção dos bins.

```bash
python -m visual_perception_experiments.fit_calibration \
  --manifest datasets/manifests/corridor-02-visual-reference.json \
  --predictions <calibration-predictions.json> \
  --out <calibration-artifact.json> \
  --source qwen_vl \
  --domain indoor_corridor
```

O artifact registra digests canônicos da referência e das predições, revisão de código,
fingerprint da configuração, contagem por claim e um `payload_digest` verificado no
carregamento. O gate padrão requer ao menos 30 decisões observadas e três bins populados.
Um artifact abaixo do gate pode existir para diagnóstico, mas carrega
`activation.enabled=false` e não é aplicado pelo pipeline.

O split `test` é usado apenas depois do ajuste, pelo relatório de avaliação já existente,
que compara raw/calibrated em ECE e Brier e também reporta cobertura, taxa de abstenção,
selective risk e casos de falha auditáveis. Nenhuma dessas métricas é produzida enquanto
a revisão humana da #210 estiver pendente.
