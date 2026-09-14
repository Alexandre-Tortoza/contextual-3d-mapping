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
- salvar mapas com contexto em pastas independentes;
- listar e comparar runs com contexto no viewer;
- validar e reparar a estrutura local.

A extração aceita amostragem por intervalo ou todos os frames. O default de
amostragem é um keyframe a cada 2 segundos. O modo integral mostra a quantidade
e o armazenamento previstos e exige que o usuário digite o `segment-id` exato.

Uma rosbag genérica pode ser inspecionada, extraída e processada por
`visual-perception`. Mapeamento e composição exigem um perfil com
calibração e configuração compatíveis; nesta versão, o perfil completo é
`corridor-02`. Sem perfil, nenhuma calibração ou máscara de outro dataset é
aplicada implicitamente. Publicar ou abrir um mapa contextual existente usa
seus próprios metadados e não executa percepção nem FAST-LIO.

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

## Salvar e comparar runs com contexto

Cada composição por `compose` salva uma nova pasta e a publica automaticamente.
Use `--run-name` para distinguir braços que reutilizam o mesmo mapa geométrico,
como uma comparação entre `no-prior` e `box-overlap`; o nome publicado preserva
o prefixo sequencial `AAAA-MM-DD-run-NNN-`.
Para importar um resultado ou uma ablação já existente:

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli publish \
  --artifact artifacts/grounding-validation/pose.json \
  --run-id grounding-run \
  --label 'Grounding + boundary + pose interpolada'

apps/cli/.venv/bin/contextual-3d-mapping-cli runs
apps/cli/.venv/bin/contextual-3d-mapping-cli serve --run-id grounding-run
```

`publish` exige um mapa contextual e seus previews locais. `--run-id` e
`--label` são opcionais: sem identidade explícita, a publicação usa data UTC e
fingerprint do mapa e das imagens. Repetir a publicação do mesmo conteúdo
reutiliza a pasta; outra versão precisa de outra identidade e nunca sobrescreve
a anterior. O nome legível também fica preservado na publicação original.

`runs --json` emite os metadados para automação. `serve` sem argumentos abre a
run publicada mais recentemente; com `--artifact`, publica antes de abrir.
O comando imprime a URL, reutiliza o servidor local deste catálogo quando
disponível ou inicia o Vite na porta 5173. O seletor **Run** permite alternar
somente entre mapas com contexto. Nas versões da mesma geometria, a câmera
permanece na posição escolhida para facilitar a comparação.

As mesmas ações estão no menu: **Publicar mapa com contexto**, **Comparar runs
com contexto** e **Servir viewer web**. Consulte também o
[contract de publicação do viewer](../map-explorer/README.md#pastas-de-runs).

## Artifacts

Para um `segment-id` chamado `<id>`, a extração publica:

```text
artifacts/<id>-window.json
artifacts/<id>-frames/*.png
```

O comando rejeita identidades que já possuam artifacts, evitando misturar
frames de execuções diferentes. Use um novo `segment-id` para repetir o fluxo.

Repetir apenas `compose` para o mesmo segmento cria outra identidade de run:

```text
artifacts/runs/<run-id>/
├── context.json
├── context-assets/
├── manifest.json
├── window.json
└── perception-manifest.json
```

O manifest da composição referencia a geometria reutilizada e a run de
percepção, além de preservar a janela e o manifest de percepção. A cópia
servida, com mapa, previews e hashes, fica em
`apps/map-explorer/web/public/runs/<run-id>/`. Ambas as raízes de resultados
são locais e ignoradas pelo Git.

## Validação

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli validate
apps/cli/.venv/bin/contextual-3d-mapping-cli validate --repair
```

`--repair` cria somente diretórios derivados ignorados pelo Git. Ele não instala
dependências, não baixa datasets e não sobrescreve artifacts.

## Selecionar a política de visibilidade

`compose` usa superfícies medidas por default e aceita ablações explícitas:

```bash
apps/cli/.venv/bin/contextual-3d-mapping-cli compose \
  --segment-id corridor-02-176s-30s \
  --window artifacts/old/integrated-grounding-6s-20260912T050652Z/window.json \
  --visual-run artifacts/old/integrated-grounding-6s-20260912T050652Z/perception/samples/20260912T050915Z \
  --visibility-mode measured_surfaces
```

Valores disponíveis: `measured_surfaces`, `dense_cells`, `legacy_cells`.
A política fica no manifest da run. A operação reutiliza as entradas existentes
e publica outra pasta comparável no viewer.
