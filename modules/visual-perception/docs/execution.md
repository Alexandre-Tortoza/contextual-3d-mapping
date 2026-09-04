# Execução

## Ciclo de vida do modelo (issue #171)

`application/lifecycle.ModelLifecycleManager` permite à orquestração criar um adapter
pesado por estágio, executá-lo, registrar `StageMetrics` (tempo de load, tempo de
inferência, pico de memória) e liberar a referência. Use-o quando a aplicação precisar
limitar modelos residentes; `run_canonical_pipeline` continua recebendo ports já
compostos e não cria um lifecycle manager implicitamente.

```python
manager = ModelLifecycleManager()
with manager.stage("region_discovery", lambda: load_real_backend(config)) as model:
    result = model.discover(image, config)
```

O pico de memória é medido como CPU-RSS neste ambiente GPU-free (uma proxy, não VRAM
real); a medição real de pico de VRAM pertence à validação experimental dos adapters.
Uma condição de out-of-memory durante load ou inferência aparece como
`domain.errors.BackendExecutionError` — nunca uma substituição silenciosa por outro
backend ou configuração.

## Cache de estágio (issue #170)

`application/cache.StageCache` persiste um registro JSON por estágio concluído,
indexado por um fingerprint encadeado (`compute_fingerprint`): o fingerprint de um
estágio faz hash de sua própria versão e configuração junto com o fingerprint de cada
estágio upstream. Consequências:

- uma execução idêntica reutiliza todo estágio em cache válido;
- mudar a configuração de um estágio invalida esse estágio e tudo que foi computado a
  partir de sua saída (seus dependentes downstream) — estágios irmãos que não dependem
  dele permanecem válidos;
- uma execução interrompida é retomável: a próxima execução com os mesmos fingerprints
  continua de onde parou;
- `CACHE_SCHEMA_VERSION` rejeita uma entrada de cache escrita por uma versão de módulo
  incompatível, em vez de reutilizá-la.

O cache é uma otimização de execução, não a serialização pública de uma observação.
Para persistir uma `VisualObservation` entre processos, use a fronteira descrita em
[artifacts.md](artifacts.md). Para escolher e configurar ports, consulte
[integration.md](integration.md).

## Verificações de qualidade

```bash
cd modules/visual-perception
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

`contextual_mapping_contracts` (e, apenas para os testes de integração,
`contextual_mapping_adapters`/`contextual_mapping_datasets`) são resolvidos a partir de
suas source trees via `[tool.pytest.ini_options].pythonpath` no `pyproject.toml`, até
que esses pacotes tenham seu próprio build instalável (veja
[model-backends.md](model-backends.md) para o gap equivalente do lado de ML).
