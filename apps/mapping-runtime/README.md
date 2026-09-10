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
ficam no diretório homônimo com sufixo `-assets`.

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
