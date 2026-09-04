# Adapters

Adapters de nível de repositório integram sistemas externos compartilhados por mais de uma capacidade ou aplicação.

Grupos iniciais de adapter:

```text
adapters/
├── datasets/
├── ros2/
└── map-storage/
```

Um adapter usado exclusivamente por um módulo deve permanecer na camada de infraestrutura desse módulo. Adapters compartilhados pertencem aqui.

Exemplos incluem normalização de dataset, tradução de fronteira ROS 2, armazenamento persistente de mapa, artifact stores, e integração de transporte externo de nível de repositório.

Adapters devem traduzir representações externas em contracts do projeto, em vez de expor schemas de terceiros diretamente aos módulos.
