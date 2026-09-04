# Dataset Adapters

Adapters compartilhados que traduzem layouts de dataset suportados para os contracts de entrada do repositório.

Parsing específico de dataset, normalização de timestamp, carregamento de calibração, nomenclatura de frame e resolução de artifact pertencem aqui quando reutilizados entre aplicações ou módulos.

`DatasetFilesystem` é a base de filesystem para um adapter concreto: recebe a raiz do
repositório e um `DatasetManifest`, exige que `datasets/raw/<dataset-id>/` exista e
resolve somente artifacts relativos que permaneçam dentro dessa raiz. Ele não faz
parsing de formato, leitura de ROS bag ou decodificação de payload.

O pacote público `contextual_mapping_adapters` expõe um protocolo `MultimodalDatasetAdapter` estreito e valores `CanonicalObservation` independentes de payload. Toda observação preserva identidade de origem, timestamp em nanossegundos inteiro, clock, frame, sensor, calibração e referência de artifact externo. Metadados de manifest inválidos falham antes da iteração.

`synchronize` cria um grupo por observação âncora configurada. A ordem de entrada é ignorada; a observação não utilizada mais próxima de cada tipo esperado é selecionada dentro de uma tolerância inclusiva. Empates são resolvidos por timestamp, índice de sequência e id de observação. Os timestamps originais são preservados e tipos ausentes são explícitos. Uma chamada aceita exatamente uma sequência e um clock; interpretação, interpolação e fusão permanecem responsabilidades downstream.
