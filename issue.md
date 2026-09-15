# CLI Interativa para o Mapeamento Contextual 3D

## Visão geral

Montar uma interface de linha de comando interativa que guie o usuário através das operações mais frequentes do projeto sem exigir memorização de comandos, paths ou flags. A CLI deve ser autodescritiva, oferecer completamento de path, validação de entrada e sugestões contextuais.

## Problema

Hoje o usuário precisa:
1. Lembrar ou consultar docstring de cada comando (`make corridor-02-window`, `make corridor-02-context`, etc);
2. Copiar e substituir paths manualmente (runs de GPU, artifacts, datasets);
3. Validar que estão usando os paths corretos e que os arquivos existem;
4. Encadear comandos manualmente.

Exemplo atual:
```bash
make corridor-02-window SEGMENT_START_S=176.3 SEGMENT_SECONDS=30 SEGMENT_ID=corridor-02-fastlio-176s-30s
cd modules/visual-perception
python benchmarks/prepare_corridor02_frames.py --window ../../artifacts/corridor-02-fastlio-176s-30s-window.json ...
.venv/bin/python benchmarks/validate_reference_pipeline.py --frame-id corridor-02-04234 ...
cd ../..
make corridor-02-context PYTHON=modules/visual-perception/.venv/bin/python M1_VISUAL_RUN=...
make map-explorer-serve
```

## Solução

Uma CLI interativa (`python -m cli` ou `contextual-3d-mapping-cli`) que:

1. **Menu principal** com as operações mais frequentes:
   - Extrair keyframes de um trecho
   - Rodar visual-perception
   - Compor artifact contextual
   - Servir o viewer web
   - Validar e reparar estrutura do projeto

2. **Fluxos guiados** (wizard):
   - "Processar um novo trecho" (paramétrico: start_s, duration_s, id)
   - "Treinar percepção em keyframes existentes"
   - "Publicar um artifact no viewer"

3. **Navegação de paths com autocompletar**:
   - Listar windows disponíveis
   - Listar runs de GPU
   - Listar datasets
   - Sugerir paths quando digitar

4. **Rosbags e intervalos**:
   - Aceitar um trecho definido por início e duração
   - Aceitar o stream RGB inteiro da rosbag
   - Permitir amostragem por intervalo ou processamento de todos os frames
   - Exigir confirmação textual para o modo integral de alto custo

5. **Validação integrada**:
   - Confirmar que o window existe antes de extrair frames
   - Avisar se visual-perception não foi rodada para um trecho
   - Verificar que o artifact foi gerado antes de servir

6. **Descrições contextuais**:
   - O que cada operação faz (uma ou duas frases)
   - Tempo estimado
   - O que será gerado (onde ficam os arquivos)
   - Requisitos (GPU? Espaço em disco?)

## Arquitetura

```text
apps/cli/
├── pyproject.toml
├── configs/
│   └── corridor-02.toml # perfil de composição conhecido
├── src/contextual_mapping_cli/
│   ├── __init__.py
│   ├── __main__.py
│   ├── main.py          # entry point, menu e subcomandos
│   ├── models.py        # tipos de requests, artifacts e profiles
│   ├── project.py       # descoberta e validação do projeto
│   ├── processes.py     # subprocessos em foreground e Ctrl+C
│   └── workflows.py     # orquestração das operações
└── tests/
    ├── test_cli.py
    ├── test_project.py
    └── test_workflows.py
```

## Componentes principais

### 1. `main.py` — Menu principal

```
Contextual 3D Mapping CLI
========================

Escolha uma operação:

 1. Extrair keyframes de um trecho
 2. Rodar visual-perception
 3. Compor artifact contextual
 4. Servir viewer web
 5. Validar estrutura do projeto
 6. Sair

> 
```

### 2. `workflows.py` — Extrair keyframes

Fluxo:
- Listar trechos já resolvidos (`artifacts/*-window.json`)
- Permitir escolher ou criar um novo
- Se novo: perguntar start_s, duration_s, SEGMENT_ID
- Rodar `make corridor-02-window` (ou equivalente parametrizado)
- Chamar `prepare_corridor02_frames.py` automaticamente
- Avisar onde os frames ficaram

### 3. `workflows.py` — Rodar visual-perception

Fluxo:
- Listar frames extraídos
- Permitir selecionar quais rodar (com multiselect)
- Avisar GPU necessária
- Rodar `validate_reference_pipeline.py` em foreground
- Mostrar progresso
- Avisar quando terminar e onde os resultados estão

### 4. `workflows.py` — Compor contextual

Fluxo:
- Listar windows resolvidos
- Listar runs de perception disponíveis
- Validar que há correspondência (window → frames extraídos → perception rodada)
- Rodar `make corridor-02-context` com os parâmetros
- Publicar no viewer (`make map-explorer-serve`)
- Abrir URL no navegador ou avisar onde acessar

### 5. `project.py` — Navegação de filesystem

- `available_bags()`: lista rosbags sob `datasets/raw/`
- `available_windows()`: lista windows resolvidas
- `available_runs()`: lista runs de GPU com manifest completo
- `available_profiles()`: carrega as composições configuradas em TOML
- Typer completa paths nos subcomandos e Questionary apresenta as opções descobertas

### 6. `project.py` — Estado do projeto

Encapsula:
- Localização dos artifacts
- Localização das runs
- Localização dos datasets
- Métodos para listar/validar cada tipo de recurso

## Bibliotecas

- **[`typer`](https://typer.tiangolo.com/)**: framework dos subcomandos
- **[`questionary`](https://questionary.readthedocs.io/)**: prompts interativos com setas e multiselect
- **[`rich`](https://rich.readthedocs.io/)**: renderização de texto colorido, tabelas, progress bars
- **`pathlib`**: navegação de filesystem (padrão)
- **`subprocess`**: rodar comandos make/python (padrão)

## Verificação

1. **Testes unitários** (`tests/`):
   - Parsing de path, validação de Window, lógica de Run
   - Sugestões de path (sem I/O real)

2. **Testes de integração**:
   - Menu principal funciona e responde
   - Comando de extract valida e roda make
   - Command de composition encadeia tudo

3. **Manual** (primeira vez que rodar):
   - Criar um novo segment (start, duration, id)
   - Extrair 3 keyframes
   - Compor e servir
   - Abrir no navegador e confirmar que aparece no seletor

## Próximas fases (fora deste plano)

- Exportar config para YAML (reusar últimas escolhas)
- Perfis completos de mapeamento para outros datasets além de corridor-02
- Integração com git (auto-commit de artifacts?)

## Critérios de sucesso

- [x] Usuário consegue fazer "novo trecho → percepção → viewer" sem consultar docs
- [x] Prompts mostram sempre o que o campo quer e onde vai ficar o resultado
- [x] Paths são sugeridos / autocompletados quando possível
- [x] Validação impede que o usuário rode comandos com entradas inválidas
- [x] Trechos e bags inteiras aceitam amostragem ou todos os frames
- [x] Tudo em português, com descrições técnicas de 1-2 linhas
