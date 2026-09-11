# Semântica de cores do viewer — mudanças implementadas

## Problema resolvido

Antes:
- **Paredes, piso, teto** (superfícies estruturais genéricas) ocupavam a maioria dos pontos
- Eles **roubavam slots de cor** (FAMILY_COLORS[0]) porque ranking era por frequência
- Objetos reais interessantes (pallets, placas, grafites, rachaduras) caíam no bucket cinza "Outros labels"
- Nenhum mecanismo para **ocultar por padrão** superfícies genéricas

Depois:
- **Categoria "Estruturas" unificada** — uma cor neutra única para wall/floor/ceiling/panel/etc.
- **Oculta por padrão** no carregamento do artifact
- Objetos reais recebem **cores distintas e estáveis** por tipo, não por frequência
- Usuário pode reativar estruturas manualmente como qualquer outra linha da legenda

## Implementação técnica

### No viewer (`apps/map-explorer/web/src/`)

1. **`map-data.js` — Classificação**
   - Novo `STRUCTURAL_HEAD_NOUNS` (wall, floor, ceiling, panel, baseboard, molding, partition, plank, etc.)
   - Sincronizado com `modules/visual-perception/domain/structural_consistency.py:INHERENTLY_STUFF_HEAD_NOUNS`
   - Função `isStructuralLabel(label)`: tokeniza, pega o núcleo (última palavra), checa se é estrutural
   - **Resultado**: cada ponto sabe se é estrutura genérica ou achado interessante

2. **`map-data.js` — Partição de paleta**
   - `buildContextPalette()` agora particiona labels em:
     - `structuralCounts` → uma única entrada `"Estruturas"` com cor neutra (RGB 150,150,150)
     - `labelCounts` → achados interessantes, agrupados por família (textual heuristic)
   - Ranking de `FAMILY_COLORS` aplica-se **só a achados**, não a estruturas
   - Famílias interessantes além da 4ª caem no bucket "Outros labels"

3. **`main.jsx` — Default oculto**
   - `defaultEnabledKeys` calcula: todas as chaves da legenda, menos `STRUCTURAL_KEY`
   - Aplicado ao abrir cada artifact (dentro de `openSlice`)
   - Aplicado ao resetar filtro (`onReset`) — "Focar tudo" agora = "estruturas ocultas + tudo mais"

### No pipeline de DEBUG

1. **Visual-Perception** (`modules/visual-perception/`)
   - Novo módulo `infrastructure/debug_recorder.py`: grava verdicts por frame
   - `partition_observation()` registra em JSON: contagem de `CONTEXT_BEARING` vs `GENERIC_STRUCTURAL_SURFACE` vs `UNINTERPRETED`
   - Arquivo: `artifacts/<segment-id>-DEBUG/visual-perception/<frame-id>.json`

2. **Mapping-Runtime** (`apps/mapping-runtime/`)
   - `corridor02_context.py` grava distribuição final de labels no artifact (ground-truth pré-agrupamento)
   - Arquivo: `artifacts/<segment-id>-DEBUG/composition.json`
   - Inclui: contagem por label, distribuição de `support_state`, exemplos de labels por suporte

3. **Viewer (Node.js)**
   - Script `apps/map-explorer/scripts/dump_palette_debug.mjs`
   - Carrega artifact e audita a paleta construída sem abrir browser
   - Comando: `node dump_palette_debug.mjs <artifact.json> [output.json]`
   - Saída: breakdown de famílias interessantes, contagem de estruturais, bucket "Outros"

## Arquivos modificados

- ✏️ `apps/map-explorer/web/src/map-data.js` — separação e cor estrutural
- ✏️ `apps/map-explorer/web/src/main.jsx` — default oculto de estruturas
- ✏️ `modules/visual-perception/src/visual_perception/application/contextual_publication.py` — log de verdicts
- ✏️ `modules/visual-perception/src/visual_perception/application/pipeline.py` — passa observation_id para debug
- ✏️ `apps/mapping-runtime/src/mapping_runtime/corridor02_context.py` — grava distribuição final
- ✨ `modules/visual-perception/src/visual_perception/infrastructure/debug_recorder.py` (novo)
- ✨ `apps/map-explorer/scripts/dump_palette_debug.mjs` (novo)
- ✏️ `.gitignore` — ignora `artifacts/*-DEBUG/`

## Como testar

### Visual (viewer)
```bash
make map-explorer-serve
# Abre um artifact — estruturas devem estar ocultas
# Legenda "Estruturas" está lá, desabilitada por padrão
# Clicar no toggle reativa; achados interessantes têm cores distintas
```

### Debug local
```bash
# Após uma execução de composição:
node apps/map-explorer/scripts/dump_palette_debug.mjs \
  artifacts/corridor-02-context.json \
  artifacts/corridor-02-DEBUG/viewer.json

# Inspeciona: JSON com breakdown de famílias, cores atribuídas, etc.
```

### Auditoria de pipeline
```bash
# Após visual-perception:
cat artifacts/corridor-02-DEBUG/visual-perception/frame-*.json
# Vê: contagem de verdicts (CONTEXT_BEARING vs GENERIC_STRUCTURAL_SURFACE)

# Após corridor02_context:
cat artifacts/corridor-02-DEBUG/composition.json
# Vê: distribuição real de labels por ponto (antes agrupamento do viewer)
```

## Notas de design

1. **Sem hardcoding de "tipos interessantes"**  
   O usuário pediu explicitamente para NÃO assumir um conjunto fechado de achados (cracks, graffiti, etc.).
   Solução: estruturas = núcleos estruturais genéricos; tudo mais = interessante.

2. **Sem mudança na fonte de dados**  
   `corridor02_context.py` continua lendo só `regions` publicadas, não `structural_context`.
   Estruturas no viewer = labels publicados cujo núcleo é genérico.

3. **Fonte de verdade sincronizada**  
   `STRUCTURAL_HEAD_NOUNS` em `map-data.js` aponta para `structural_consistency.py`.
   Comentário bloca se divergirem.

4. **DEBUG em todas as etapas**  
   Calibração: visual-perception verdicts, sensor-association/fusão (corroboração), composição final, viewer.
   Permite inspecionar fluxo sem re-rodar.
