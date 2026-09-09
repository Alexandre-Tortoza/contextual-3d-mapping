# M1 — Primeiro slice RGB–LiDAR 3D

Este workflow produz um artifact JSON que o `map-explorer` abre localmente.
Dados de origem não entram no Git: coloque a rosbag e seus arquivos de
calibração em `datasets/raw/<nome-do-dataset>/`, preservando o layout original.

## Smoke test offline

Antes de preparar ROS ou dados reais, valide toda a composição Python e o
contrato entregue ao viewer:

```bash
make m1-test
make m1-demo
make map-explorer-install map-explorer-build
```

Isso cria `artifacts/m1-demo.json` com pontos associados, um ponto ocluído,
cores, região e proveniência. O comando não afirma validar FAST-LIO: ele isola
e testa o restante do slice enquanto rosbag e calibração reais não existem.

Para o dataset local `corridor-02`, execute um trecho real de 20 segundos e
produza uma amostra geométrica pronta para o viewer:

```bash
make corridor-02-map
```

O workflow usa a configuração Velodyne VLP-16 e os extrínsecos fornecidos pelo
dataset. Ele gera `artifacts/corridor-02-fastlio-20s.pcd` com o mapa completo do
trecho e `artifacts/corridor-02-fastlio-20s.json` com no máximo 25 mil pontos.
As cores do segundo artifact codificam altura somente para inspeção; elas não
representam associação RGB–LiDAR.

## Ambiente reproduzível

Os perfis Docker isolam ROS e FAST-LIO do ambiente do host:

```bash
docker compose --profile ros1 build fastlio-ros1
docker compose --profile ros1 run --rm fastlio-ros1
docker compose --profile ros2 build fastlio-ros2
docker compose --profile ros2 run --rm fastlio-ros2
```

ROS 2/Humble é o perfil principal para sensores suportados por
`livox_ros_driver2` e usa `Livox-SDK2`. O perfil ROS 1 mantém compatibilidade
com o FAST-LIO upstream original; por isso compila `livox_ros_driver` e o
`Livox-SDK` legado dentro da imagem, sem expor essas bibliotecas aos módulos
Python.

As revisões de FAST-LIO, drivers e SDKs são pinadas nos Dockerfiles. Dentro do
container, carregue o workspace correspondente antes de executar launch files:

```bash
# ROS 1
source /opt/fast-lio/devel/setup.bash

# ROS 2
source /opt/fast-lio/install/setup.bash
```

O perfil `gpu` é opcional e só é necessário para os backends reais de
`visual-perception`:

```bash
docker compose --profile gpu run --rm visual-gpu
```

Cada ambiente ROS deve receber a rosbag e a configuração FAST-LIO específica
do sensor. `FastLioProcessAdapter` define o protocolo JSON de integração para
uma ponte externa: envia pontos LiDAR, todas as amostras IMU e metadados, e
exige pose mais pontos deskewed na resposta. A implementação ROS dessa ponte
ainda depende dos tópicos e tipos de mensagem do dataset escolhido; nenhuma
mensagem ROS atravessa a fronteira pública de `state-estimation`.

## Verificação do artifact

Depois que `mapping-runtime` exportar o slice, execute o cliente web:

```bash
cd apps/map-explorer/web
npm ci
npm run dev
```

Abra o JSON produzido no seletor de arquivo. Um ponto selecionado deve exibir
coordenadas de mapa, observação LiDAR, observação RGB, pixel, cor, região,
claim visual, feature reference e artifact de calibração quando a associação
for válida.

## Limites de M1

O slice preserva claims brutos por ponto, mas não faz normalização de labels,
semantic fusion temporal, semantic memory, scene graph ou query engine.
