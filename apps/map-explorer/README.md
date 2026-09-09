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

## Execução local

Instale dependências de forma determinística, produza o bundle e inicie o
servidor local:

```bash
make map-explorer-install map-explorer-build
cd apps/map-explorer/web
npm run dev
```

O viewer valida a versão e os campos mínimos do artifact antes de renderizar.
Para uma entrada conhecida, gere `artifacts/m1-demo.json` com `make m1-demo` e
abra esse arquivo no seletor da página.

Para visualizar um trecho real, execute `make corridor-02-map` na raiz e abra
`artifacts/corridor-02-fastlio-20s.json`. O viewer enquadra automaticamente os
limites da nuvem e permite órbita, zoom e inspeção de cada ponto amostrado.

Em uma sessão SSH, publique o artifact pelo próprio servidor para evitar o
seletor de arquivos do computador cliente:

```bash
make map-explorer-serve
```

Depois acesse `http://<ip-do-servidor>:5173/?artifact=/current-map.json`. O
arquivo publicado é local e ignorado pelo Git.
