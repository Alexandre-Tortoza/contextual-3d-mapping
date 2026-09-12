# Visual Perception

`visual-perception` transforma uma observação de imagem RGB canônica em uma observação
visual estruturada e auditável: regiões descobertas com masks e boxes, embeddings de
região densos e alinhados à linguagem, claims semânticos em nível de cena e de região,
relações candidatas em nível de imagem, e um audit de qualidade — com proveniência de
modelo e configuração anexada em todo o processo.

> Documentação detalhada: [`docs/`](docs/README.md).

O módulo é testável de forma independente e cada estágio canônico é substituível por
trás de um port, e cada estágio contextual é desligável por configuração — para que uma
comparação consiga atribuir um efeito a um estágio só. Ele vem com fakes completos,
determinísticos e GPU-free para cada backend, para que seus contracts, pipeline, cache e
fronteiras de integração possam ser totalmente exercitados sem uma GPU ou download de
modelo; backends reais são rastreados separadamente (veja
[docs/model-backends.md](docs/model-backends.md)).

## Objetivo semântico

O produto semântico do módulo deve priorizar **evidência contextual**, não classificação
estrutural trivial.

Uma parede, piso ou teto podem continuar existindo como regiões geométricas descobertas
pelo pipeline, mas não devem dominar a observação apenas por serem superfícies. O que deve
chegar ao mapa contextual são entidades e evidências que acrescentam informação, como
portas, placas, objetos, pichações, textos, símbolos, rachaduras, danos estruturais,
umidade, tinta descascada, hazards e obstruções.

A política normativa, os exemplos e o estado de adoção estão em
[docs/contextual-semantics.md](docs/contextual-semantics.md).

A implementação em `main` ainda está em transição para essa política; runs anteriores
continuam sendo baselines reproduzíveis da política semântica antiga.

## Responsabilidades

- consumir observações RGB canônicas emitidas por `[adapters]` (não ler datasets/ROS
  bags diretamente);
- descobrir, dividir em tiles e mesclar regiões de imagem em regiões canônicas estáveis;
- fazer pooling de features visuais densas e produzir embeddings alinhados à linguagem
  por região;
- interpretar semântica em nível de cena e de região como claims auditáveis, não labels
  únicos;
- distinguir geometria visual de evidência contextual, sem exigir que toda região
  geométrica receba uma identidade semântica de alto nível;
- medir, por um canal **independente** do reasoner, se a evidência sustenta cada hipótese
  de identidade;
- reinterpretar seletivamente as regiões que têm uma razão explícita, sempre com evidência
  nova;
- reconciliar, dentro do frame, conceitos equivalentes e regiões que provavelmente são a
  mesma superfície, sem apagar nenhuma delas;
- gerar relações 2D candidatas entre regiões, geométricas e semânticas, mantidas
  distinguíveis;
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
mise trust ../../.mise.toml  # uma vez, se o ambiente usa mise
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,bench]"
cd ../..
make verify
```

O alvo `verify` da raiz é a fonte única para a suíte ampliada: testes unitários, de
integração e de contract dos benchmarks, `ruff` no repositório inteiro e `mypy` em
modo estrito. Dependências pesadas dos backends reais continuam no extra `ml` e não
são necessárias para a suíte determinística em CPU.

`contextual_mapping_contracts` (e, apenas para os testes de integração,
`contextual_mapping_adapters`/`contextual_mapping_datasets`) são resolvidos a partir de
suas source trees via `pythonpath` do pytest no `pyproject.toml`, até que esses pacotes
tenham seu próprio build instalável.
