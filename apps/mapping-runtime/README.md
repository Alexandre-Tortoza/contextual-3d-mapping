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

Com o dataset `corridor-02` em `datasets/raw`, o alvo abaixo executa os primeiros
20 segundos da rosbag no FAST-LIO ROS 1 e converte o PCD para uma amostra que o
viewer consegue carregar:

```bash
make corridor-02-map
```

O resultado fica em `artifacts/corridor-02-fastlio-20s.json`. A amostragem é
determinística e mantém no máximo 25 mil pontos; `display_color_rgb` representa
somente altura geométrica e não é tratado como associação RGB.

## Associação contextual do corridor-02

Quando uma execução real de `visual-perception` está disponível, o runtime pode
associar seu primeiro frame ao scan LiDAR sincronizado e ao mapa FAST-LIO:

```bash
make corridor-02-context \
  PYTHON=modules/visual-perception/.venv/bin/python \
  M1_VISUAL_RUN=modules/visual-perception/benchmarks/results/samples/<run-id>
```

O artifact `artifacts/corridor-02-fastlio-20s-context.json` preserva os pontos
geométricos não observados e adiciona somente os pontos com projeção MEI e RGB
válidos no hemisfério frontal da câmera. A restrição é necessária porque a
equação MEI também admite raios traseiros matematicamente projetáveis, embora
eles não pertençam ao campo de visão físico deste rig. Claims de região e de
cena são mantidos como predições VLM com estado
de suporte e proveniência; não são promovidos a ground truth. Os previews ficam
no diretório homônimo com sufixo `-assets`.
