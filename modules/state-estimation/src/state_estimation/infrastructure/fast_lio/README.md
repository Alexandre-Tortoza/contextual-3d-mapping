# Adapter FAST-LIO

Este diretório contém a integração externa inicial de odometria LiDAR-inercial.

Sua responsabilidade é a tradução entre o runtime externo e os ports de state-estimation. Mensagens ROS, configuração de backend, ciclo de vida de processo, nomes de tópico e detalhes específicos de dependência devem permanecer aqui e não devem vazar para o domain ou para os contracts de módulos downstream.

`FastLioProcessAdapter` executa um comando configurado e troca um documento
JSON pela entrada padrão e saída padrão. O request inclui a referência e os
pontos do scan LiDAR, além de referências, velocidade angular e aceleração de
cada amostra IMU. A resposta deve conter:

```json
{
  "translation_m": [0.0, 0.0, 0.0],
  "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
  "points_m": [[0.0, 0.0, 1.0]]
}
```

`points_m` representa a nuvem deskewed, ainda no frame LiDAR declarado. O
adapter valida timeout, exit code, JSON, pose, pontos e ordenação temporal das
amostras IMU. A ponte ROS concreta é configuração de deployment: deve publicar
nos tópicos do FAST-LIO escolhido e traduzir suas saídas, sem introduzir tipos
ROS na API pública do módulo.
