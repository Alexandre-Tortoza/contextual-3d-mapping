# Adapter FAST-LIO

Este diretório contém a integração externa inicial de odometria LiDAR-inercial.

Sua responsabilidade é a tradução entre o runtime externo e os ports de state-estimation. Mensagens ROS, configuração de backend, ciclo de vida de processo, nomes de tópico e detalhes específicos de dependência devem permanecer aqui e não devem vazar para o domain ou para os contracts de módulos downstream.
