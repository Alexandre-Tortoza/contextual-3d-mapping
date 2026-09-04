# Map Explorer

`map-explorer` é a principal aplicação voltada a humanos para abrir, consultar e inspecionar visualmente mapas persistentes.

## Responsabilidades

- abrir um mapa através de contracts de aplicação estáveis;
- renderizar geometria 3D e overlays semânticos;
- submeter consultas semânticas, espaciais, relacionais e contextuais através do `query-engine`;
- focar o viewer nas entidades ou regiões retornadas;
- expor observações relacionadas, evidência, confiança e proveniência;
- visualizar relações de scene-graph sem possuir a construção do grafo.

## Estrutura inicial

```text
map-explorer/
├── README.md
├── api/
├── web/
└── configs/
```

A API e o frontend são camadas de entrega. Não devem depender diretamente de implementações privadas de módulo nem de schemas específicos de armazenamento.
