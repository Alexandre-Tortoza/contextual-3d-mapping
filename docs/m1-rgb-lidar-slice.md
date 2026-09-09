# M1 — Primeiro slice RGB–LiDAR 3D

Este workflow produz um artifact JSON que o `map-explorer` abre localmente.
Dados de origem não entram no Git: coloque a rosbag e seus arquivos de
calibração em `datasets/raw/<nome-do-dataset>/`, preservando o layout original.

## Ambiente reproduzível

Os perfis Docker isolam ROS e FAST-LIO do ambiente do host:

```bash
docker compose --profile ros1 build fastlio-ros1
docker compose --profile ros1 run --rm fastlio-ros1
docker compose --profile ros2 build fastlio-ros2
docker compose --profile ros2 run --rm fastlio-ros2
```

O perfil `gpu` é opcional e só é necessário para os backends reais de
`visual-perception`:

```bash
docker compose --profile gpu run --rm visual-gpu
```

Cada ambiente ROS deve receber a rosbag, a configuração FAST-LIO e a ponte
JSON-lines configurada para `FastLioProcessAdapter`. A ponte traduz as
mensagens ROS para o contract público; nenhuma mensagem ROS atravessa a
fronteira de `state-estimation`.

## Verificação do artifact

Depois que `mapping-runtime` exportar o slice, execute o cliente web:

```bash
cd apps/map-explorer/web
npm install
npm run dev
```

Abra o JSON produzido no seletor de arquivo. Um ponto selecionado deve exibir
coordenadas de mapa, observação LiDAR, observação RGB, pixel, cor, região,
claim visual, feature reference e artifact de calibração quando a associação
for válida.

## Limites de M1

O slice preserva claims brutos por ponto, mas não faz normalização de labels,
semantic fusion temporal, semantic memory, scene graph ou query engine.
