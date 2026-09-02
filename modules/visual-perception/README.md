# Image Context

Laboratorio independente para experimentar enriquecimento contextual de imagens antes de
portar os resultados para o `vlm-context-map`.

> Documentacao detalhada: [`docs/`](docs/README.md).

## Estado atual

As pipelines de producao `baseline` (VLM tres passagens -> Grounding DINO -> SAM2) e
`region-first` (SAM2 automatico -> DINOv2 -> Qwen global/local) foram removidas. Elas
existiam como experimentos paralelos e serao substituidas por um unico modulo canonico,
open-vocabulary e uncertainty-aware, desenhado no epic
[#3](https://github.com/Alexandre-Tortoza/image-context/issues/3) e suas issues filhas
(#5 a #14). O codigo removido permanece recuperavel no historico do git caso seja
necessario para comparacao cientifica futura (issue #12).

Hoje a CLI expoe apenas amostragem reproduzivel de imagens de um ROS bag, sem carregar
nenhum modelo pesado.

## Preparacao

O projeto requer Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Se o `python3.12` estiver sendo gerenciado pelo mise:

```bash
mise use python@3.12.14
python -m venv .venv
```

## Amostragem

```bash
image-context sample --config config.yaml --overwrite
```

Para conferir reproducibilidade ou testar um caso pequeno:

```bash
image-context sample --config config.yaml \
  --sample-size 1 --seed 42 --run-id smoke-sample --overwrite
```

## Saida

```text
runs/<run-id>/
├── manifest.json
├── selected_frames.json
└── frames/
    └── frame-XXXXXX/
        └── image.png
```

## Qualidade

```bash
pytest
ruff check .
mypy
```
