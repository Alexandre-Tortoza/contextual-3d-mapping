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
