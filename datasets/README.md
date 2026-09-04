# Datasets

Este diretório é dono das convenções do repositório para arquivos de dataset locais e dos metadados necessários para experimentos reprodutíveis.

```text
datasets/
├── raw/
│   └── <dataset-name>/
│       └── ...
├── manifests/
├── schemas/
├── splits/
└── contextual_mapping_datasets/
```

## Layout de dataset bruto

Arquivos de dataset baixados ou extraídos devem viver em:

```text
datasets/raw/<dataset-name>/*
```

Cada dataset recebe seu próprio diretório e deve preservar o layout upstream do dataset sempre que praticável. Adapters de dataset devem ler a partir dessa raiz de dataset em vez de espalhar arquivos de origem pelo repositório.

O identificador `DatasetManifest.dataset_id` também é o nome desse diretório. A API
`contextual_mapping_datasets.raw_dataset_root(repository_root, dataset_id)` é a forma
canônica de resolver a raiz; ela aceita apenas um segmento de caminho para impedir que
um manifest faça um adapter ler fora de `datasets/raw/`.

Exemplos:

```text
datasets/raw/cerberus-subt/...
datasets/raw/grandtour/...
datasets/raw/tartanground/...
```

`datasets/raw/` é dado de trabalho local. Seu conteúdo é intencionalmente excluído do Git porque imagens RGB, point clouds LiDAR, ROS bags, arquivos compactados e artifacts de origem similares podem ser muito grandes.

## Suporte a dataset versionado

Use os diretórios restantes para informação rastreada pelo Git:

- `manifests/` descreve identidade do dataset, sequências, fontes de sensor, clocks, frames, referências de calibração, proveniência e localizações de origem locais.
- `schemas/` contém schemas e definições de validação relacionados a dataset.
- `splits/` contém splits reproduzíveis de treino, validação, teste e avaliação.
- `contextual_mapping_datasets/` fornece o modelo de manifest versionado usado por adapters de dataset e experimentos.

A versão atual do schema de manifest, `1.0`, descreve identidade de dataset e sequência, fontes de sensor externas, clocks, frames, referências de calibração, e membership opcional em split de avaliação.

## Manifests JSON

Manifests rastreados usam JSON e ficam em `datasets/manifests/<dataset-id>.json`.
Seus `artifact_uri` são caminhos relativos à raiz canônica do dataset, nunca caminhos
absolutos ou URLs. Use `load_dataset_manifest(path)` para ler e validar o documento;
o adapter resolve cada artifact a partir de `datasets/raw/<dataset-id>/`.

Mapas gerados, checkpoints de modelo, caches e saídas de experimento não são arquivos de dataset brutos e devem permanecer em seu runtime, experimento, ou local configurado de armazenamento de artifact.
