# Mapping Runtime

`mapping-runtime` é o composition root para construir ou atualizar mapas a partir de sensores ao vivo, sessões gravadas, ou adapters de dataset.

Possui a configuração, o wiring de dependências, o ciclo de vida e a ordem de execução. Não implementa capacidades de pesquisa que pertencem aos módulos.

Estrutura inicial:

```text
mapping-runtime/
├── README.md
├── configs/
└── src/
```

O runtime deve consumir contracts públicos e pontos de entrada de módulo, para que o mesmo pipeline downstream possa operar com sensores ao vivo, dados gravados, datasets, saída de simulador ou fixtures de teste.

## Slice executável do M1

O comando `demo` compõe os contracts públicos de state-estimation, o mapa
geométrico, a associação calibrada RGB–LiDAR e a exportação atômica. Ele usa
uma fixture sintética pequena para validar o caminho completo sem mascarar a
ausência de dados reais:

```bash
make m1-demo
```

Por default, o resultado fica em `artifacts/m1-demo.json`. Para escolher outro
caminho:

```bash
make m1-demo M1_ARTIFACT=/tmp/meu-slice.json
```

O entry point Python equivalente é:

```bash
PYTHONPATH="contracts:modules/state-estimation/src:modules/geometric-map/src:modules/sensor-association/src:apps/mapping-runtime/src" \
  python -m mapping_runtime demo --output artifacts/m1-demo.json
```

O demo não substitui validação com rosbag. Sua função é provar que composição,
frames, sincronização, oclusão, cor, região e proveniência chegam a um artifact
que o viewer consegue abrir.

## Trecho FAST-LIO real

Com o dataset `corridor-02` em `datasets/raw`, o alvo abaixo resolve a janela do
trecho, executa esse trecho no FAST-LIO ROS 1 e converte o PCD para uma amostra
que o viewer consegue carregar:

```bash
make corridor-02-map
```

A janela é parametrizada:

```bash
make corridor-02-map SEGMENT_START_S=176.3 SEGMENT_SECONDS=30 \
  SEGMENT_ID=corridor-02-fastlio-176s-30s
```

`SEGMENT_START_S` conta segundos a partir do **primeiro frame RGB do bag**.
Ao mudá-lo, mude também `SEGMENT_ID`: ele nomeia os artifacts e vira o `map_id`,
e dois trechos com o mesmo identificador colidiriam no namespace de
`geometry_id`.

### Os dois relógios do bag

O bag do corridor-02 preserva dois relógios diferentes: o **tempo de gravação**
(epoch de quando o arquivo foi escrito) e o **header** de cada mensagem, que é o
relógio original da captura e o mesmo usado pelo ground-truth. `rosbag play
--start` entende apenas o primeiro; a associação e a trajetória usam o segundo.
A diferença entre eles não é constante: ela deriva dezenas de milissegundos ao
longo da sequência.

Por isso a janela é resolvida antes da execução, lendo headers reais:

```bash
make corridor-02-window
```

O resultado, `artifacts/<segment-id>-window.json`, publica o offset de
reprodução e a identidade completa de cada keyframe — posição no stream RGB,
timestamp de header e timestamp de gravação —, que é o que a composição
contextual precisa para reencontrar cada frame.

O trecho começa `SEGMENT_LEAD_S` segundos antes da janela pedida, para o
estimator inercial convergir antes do intervalo que será mapeado.

O comando também aceita uma rosbag inteira e pode selecionar todas as imagens:

```bash
python -m mapping_runtime bag-window \
  --bag datasets/raw/<dataset>/<run>.bag \
  --whole-bag \
  --all-frames \
  --camera-topic /camera/image_raw \
  --recording-id <run> \
  --output artifacts/<run>-window.json
```

`recording_id` e `frame_id_prefix` preservam a identidade da gravação nos
artifacts. Leitores continuam aceitando windows antigas que não possuem esses
campos, usando `corridor-02` como prefixo compatível.

## Contexto de um trecho

Extraia os keyframes da janela, rode `visual-perception` sobre eles e componha:

```bash
cd modules/visual-perception
python benchmarks/prepare_corridor02_frames.py \
  --window ../../artifacts/<segment-id>-window.json \
  --out-dir benchmarks/.local/corridor-02-frames
python benchmarks/validate_reference_pipeline.py --frame-id corridor-02-04234 ...
cd ../..
make corridor-02-context \
  PYTHON=modules/visual-perception/.venv/bin/python \
  M1_VISUAL_RUN=modules/visual-perception/benchmarks/results/samples/<run-id>
```

Os PNGs extraídos por janela são nomeados pelo índice original do frame no bag
(`corridor-02-04234.png`), e não pela ordem da amostragem. Essa identidade é o
que liga a execução de percepção de volta ao keyframe e à sua pose.

O artifact contextual **rotula os próprios pontos do mapa**: ele não anexa o
scan colorido ao lado da geometria acumulada. Cada ponto persistido termina com
um label ou com o motivo explícito de não ter um — fora do campo de visão,
projetado fora da imagem, fora do suporte óptico válido, ou ocluído por uma
superfície mais próxima.

A restrição ao hemisfério frontal continua necessária porque a equação MEI
também admite raios traseiros matematicamente projetáveis, embora eles não
pertençam ao campo de visão físico deste rig.

Quando vários keyframes classificam o mesmo ponto, `semantic-fusion` escolhe o
label primário e registra a concordância, preservando todos os contribuintes.
Claims de região e de cena continuam sendo predições VLM com estado de suporte e
proveniência; não são promovidos a ground truth. Os previews de cada keyframe
ficam em `assets/`, dentro da própria pasta da run.

O alvo Make `corridor-02-context` grava por default em
`artifacts/runs/<AAAA-MM-DD>-run-<número>-<segment-id>/context.json`, preservando
versões do mesmo segmento. Por exemplo, `2026-09-12-run-010-corridor-02`. O
número sequencial absoluto facilita identificar a última run no catálogo local.
`M1_CONTEXT_ARTIFACT` permite escolher o destino explicitamente.
A [CLI](../cli/README.md#salvar-e-comparar-runs-com-contexto) também preserva
uma cópia da janela e do manifest de percepção em cada composição e publica
automaticamente o resultado para comparação no viewer. O publisher de
`map-explorer` mantém sua própria pasta imutável com mapa, previews e hashes.

### Fonte de pose

A pose que registra o mapa na câmera vem, por ordem de preferência:

1. da odometria gravada durante a execução do FAST-LIO
   (`artifacts/<segment-id>-odometry.csv`), que descreve exatamente a trajetória
   que originou o mapa;
2. do ground-truth do dataset, realinhado ao instante em que o mapa começou.

A primeira é preferida porque, com o contexto ancorado nos pontos do mapa, um
erro de pose entra direto na projeção: vira pixel errado e, portanto, label
errado. O ground-truth permanece como alternativa auditável, e é aproximado —
ele não compartilha a origem nem o alinhamento gravitacional do FAST-LIO.

A amostragem padrão interpola translação e rotação para o timestamp RGB exato.
O runtime rejeita extrapolação e intervalos maiores que `max_pose_gap_ns`
(200 ms por default). `--pose-sampling nearest` preserva a seleção histórica
para ablação; a proveniência registra qual modo e quais amostras foram usados.
O realinhamento inicial do ground-truth permanece igual ao anterior.

### Footprints e comparação espacial

`corridor-02-context` consome `ObservedRegion.grounding` por default. Artifacts
visuais antigos continuam legíveis, mas não possuem grounding e não autorizam
labels fortes. Claims, structural context e falhas continuam preservados.
`--footprint-mode legacy_discovery` é uma ablação explícita do comportamento
histórico. O runtime não carrega modelos: o grounding é produzido previamente
por `visual-perception`, apenas nas candidatas a publicação contextual.

`--disable-boundary-policy` isola o grounding; `--registration-sigma-px` informa
incerteza adicional em pixels. `--stuff-discovery-fallback` permite footprint
de discovery somente como evidência tentativa para `stuff`. Nunca é fallback
forte para `thing`, `part` ou natureza desconhecida.

Cada região registra discovery, predição do segmentador, máscara semântica e
RLE da máscara após ownership, inclusive componentes rejeitados e falhas.
Cada ponto mantém a distância à boundary, a margem e a referência ao grounding.
As contribuições multi-frame preservam seu próprio pixel, para que uma projeção
possa ser inspecionada sem usar o pixel da observação vencedora por engano.

`experiments/visual_perception_experiments/grounding_validation.py` reexecuta
somente grounding sobre observações congeladas de uma janela de até dez
segundos e exporta braços independentes de discovery, grounding, boundary e
pose, com comparações 2D/3D e métricas. Não gera outro mapa nem repete o VLM.

## Geometria integral e visibilidade

`corridor-02-context` usa `--visibility-mode measured_surfaces` por default.
Carrega o PCD completo em `geometric_slice.source.uri` e verifica
`source.sha256` antes de associar os pontos de exibição. Quando a origem foi
movida, `--visibility-geometry /caminho/mapa.pcd` indica seu novo caminho; o
hash precisa continuar igual. Ausência de PCD ou digest é erro acionável.

`--visibility-mode legacy_cells` e `--visibility-mode dense_cells` selecionam
os braços da comparação ativa. Pelo Make, use
`M1_VISIBILITY_MODE=measured_surfaces` (default). Recompor reutiliza mapa,
trajetória, imagens e percepção; não executa FAST-LIO nem inferência visual.

O artifact registra `visibility` com configuração, origem, contagem de pontos
e patches; cada observação registra `visibility_counts`; regiões registram
`surface_support`. Pontos, contribuições e hipóteses tentativas preservam
`surface_evidence`. Geometria rejeitada continua no mapa com motivo explícito.
A opção de composição `audit_geometry_ids` grava o resultado por frame dos IDs
solicitados em `debug/sensor-association/`, para regressões auditáveis.

Algoritmo e limitações pertencem a
[`sensor-association`](../../modules/sensor-association/docs/measured-visibility.md).
