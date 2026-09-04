# Visual Perception

`visual-perception` transforma uma observação de imagem RGB canônica em uma observação
visual estruturada e auditável: regiões descobertas com masks e boxes, embeddings de
região densos e alinhados à linguagem, claims semânticos em nível de cena e de região,
relações candidatas em nível de imagem, e um audit de qualidade — com proveniência de
modelo e configuração anexada em todo o processo.

> Documentação detalhada: [`docs/`](docs/README.md).

O módulo é testável de forma independente e cada estágio canônico é substituível por
trás de um port. Ele vem com fakes completos, determinísticos e GPU-free para cada
backend, para que seus contracts, pipeline, cache e fronteiras de integração possam ser
totalmente exercitados sem uma GPU ou download de modelo; backends reais são rastreados
separadamente (veja [docs/model-backends.md](docs/model-backends.md)).

## Responsabilidades

- consumir observações RGB canônicas emitidas por `[adapters]` (não ler datasets/ROS
  bags diretamente);
- descobrir, dividir em tiles e mesclar regiões de imagem em regiões canônicas estáveis;
- fazer pooling de features visuais densas e produzir embeddings alinhados à linguagem
  por região;
- interpretar semântica em nível de cena e de região como claims auditáveis, não labels
  únicos;
- gerar relações 2D candidatas entre regiões;
- auditar a observação resultante quanto à consistência estrutural e contradições;
- fazer cache de estágios caros e serializar a observação canônica para persistência.

## Não-responsabilidades

- sampling e transporte de dataset/ROS bag (`[adapters]`);
- calibração, projeção cross-sensor, associação com LiDAR (`sensor-association`);
- construção de mapa geométrico/semântico persistente, scene graphs.

## Estrutura

```text
visual-perception/
├── README.md
├── docs/
├── benchmarks/
├── src/
│   └── visual_perception/
│       ├── domain/
│       ├── ports/
│       ├── application/
│       ├── infrastructure/
│       │   ├── fakes/
│       │   ├── adapters/
│       │   └── integration/
│       └── config.py
└── tests/
```

## Desenvolvimento

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
mypy
```

`contextual_mapping_contracts` (e, apenas para os testes de integração,
`contextual_mapping_adapters`/`contextual_mapping_datasets`) são resolvidos a partir de
suas source trees via `pythonpath` do pytest no `pyproject.toml`, até que esses pacotes
tenham seu próprio build instalável.
