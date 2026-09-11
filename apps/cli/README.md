# CLI

`cli` guia os workflows operacionais do mapeamento contextual sem exigir que o
usuário memorize comandos, paths e variáveis do Makefile. A mesma composição é
oferecida por menu e por subcomandos reproduzíveis.

## Instalação

A partir da raiz do repositório:

```bash
python -m venv apps/cli/.venv
apps/cli/.venv/bin/python -m pip install -e 'apps/cli[dev]'
```

Abra o menu interativo:

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli
```

Também é possível usar `apps/cli/.venv/bin/python -m contextual_mapping_cli`.

## Workflows

O menu principal permite:

- processar um trecho ou uma rosbag inteira;
- executar `visual-perception` sobre frames existentes;
- compor um artifact contextual;
- publicar e servir o viewer;
- validar e reparar a estrutura local.

A extração aceita amostragem por intervalo ou todos os frames. O default de
amostragem é um keyframe a cada 2 segundos. O modo integral mostra a quantidade
e o armazenamento previstos e exige que o usuário digite o `segment-id` exato.

Uma rosbag genérica pode ser inspecionada, extraída e processada por
`visual-perception`. Mapeamento, composição e viewer exigem um perfil com
calibração e configuração compatíveis; nesta versão, o perfil completo é
`corridor-02`. Sem perfil, nenhuma calibração ou máscara de outro dataset é
aplicada implicitamente.

## Uso não interativo

Trecho amostrado:

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli extract \
  --bag datasets/raw/corridor-02/corridor-02.bag \
  --segment-id corridor-02-176s-30s \
  --start-s 176.3 \
  --duration-s 30 \
  --interval-s 2
```

Rosbag inteira, amostrada:

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli process \
  --bag datasets/raw/corridor-02/corridor-02.bag \
  --segment-id corridor-02-full \
  --whole-bag \
  --interval-s 2
```

Todos os frames exigem confirmação forte:

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli extract \
  --bag datasets/raw/corridor-02/corridor-02.bag \
  --segment-id corridor-02-full-all \
  --whole-bag \
  --all-frames \
  --confirm-all-frames corridor-02-full-all
```

Os processos longos rodam em foreground, transmitem seus logs e são encerrados
de forma segura com `Ctrl+C`.

## Artifacts

Para um `segment-id` chamado `<id>`, a extração publica:

```text
artifacts/<id>-window.json
artifacts/<id>-frames/*.png
```

O comando rejeita identidades que já possuam artifacts, evitando misturar
frames de execuções diferentes. Use um novo `segment-id` para repetir o fluxo.

## Validação

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli validate
apps/cli/.venv/bin/contextual-3d-mapping-cli validate --repair
```

`--repair` cria somente diretórios derivados ignorados pelo Git. Ele não instala
dependências, não baixa datasets e não sobrescreve artifacts.
