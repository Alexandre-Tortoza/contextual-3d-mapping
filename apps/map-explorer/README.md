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
make map-explorer-install map-explorer-test map-explorer-build
cd apps/map-explorer/web
npm run dev
```

O viewer valida a versão e os campos mínimos do artifact antes de renderizar.
Ele abre `/current-map.json` por default e aceita outra URL pelo parâmetro
`?artifact=`.

O seletor compacto no mapa alterna entre o artifact contextual atual — um
frame visual sobre a geometria acumulada — e o trecho puramente geométrico de
20 segundos. Essa distinção evita apresentar contexto de um frame como se já
fosse fusão temporal de todo o trecho.

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

Artifacts contextuais expõem geometria, associações RGB e claims do VLM. O
mapa usa uma única camada de cores contextuais; pontos sem label permanecem em
cinza discreto para preservar a referência geométrica. Ao selecionar um ponto
observado, o painel mostra o frame de origem,
o pixel projetado, a região, a confiança, o estado de suporte e a proveniência.
Pontos sem observação visual permanecem explicitamente sem contexto. O alvo
`map-explorer-serve` usa por default o artifact contextual e copia seus previews;
outro arquivo pode ser selecionado com `MAP_EXPLORER_ARTIFACT=artifacts/outro.json`.

## Navegação

O modo `Explorar` usa botão esquerdo para orbitar, botão direito ou central
para mover câmera e alvo, e scroll para aproximar no cursor. `Modo voo` adiciona
`W/A/S/D` para movimento horizontal, `Q/E` para altura e `Shift` para acelerar.
Seleção não move a câmera; duplo clique ou `F` foca o ponto. `Home` restaura a
vista geral, enquanto `1` e `2` abrem as vistas superior e isométrica.

A sidebar `Detalhes` concentra a inspeção. A legenda contextual fica em um
painel pequeno e recolhível no canto inferior esquerdo do mapa e permite ocultar
ou isolar labels sem recarregar a geometria. O frontend consome a nuvem por uma
fronteira interna compatível com bounds e LOD futuros; o artifact local atual
continua sendo servido como um único chunk estático.
