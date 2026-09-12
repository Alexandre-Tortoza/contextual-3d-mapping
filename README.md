# contextual-3d-mapping

Framework de pesquisa para construir mapas 3D semânticos e contextuais de vocabulário aberto a partir de RGB, LiDAR e estimativas de movimento, combinando geometria persistente do mundo, features visual-language, representações de pontos aprendidas, memória semântica, grafos de cena e raciocínio espacial.

## Arquitetura

O repositório usa uma **arquitetura modular simples orientada a capacidades**.

Os principais objetivos são legibilidade, responsabilidade explícita, baixo acoplamento, testabilidade, substituibilidade, manutenibilidade e reprodutibilidade científica. Os princípios SOLID guiam o design de código e dependências nas fronteiras relevantes.

A regra central é simples:

```text
uma capacidade -> um módulo dono claro -> API pública pequena -> composição explícita
```

Integrações e detalhes de implementação específicos de capacidade ficam com o módulo dono. Aplicações compõem módulos em workflows executáveis. Primitivas compartilhadas permanecem pequenas e estáveis.

## Fluxo de alto nível

```text
RGB + LiDAR + IMU
        -> state-estimation
        -> geometric-map
        -> visual-perception / point-representation
        -> sensor-association
        -> semantic-fusion
        -> semantic-map
        -> semantic-memory / scene-graph / context-reasoning
        -> query-engine
        -> applications
```

A documentação didática que acompanha esse fluxo dado a dado, de `ImageObservation`/`ImagePayload` até `PointVisualAssociation`, `SemanticContribution` e `FusedPointContext`, está em [`docs/end-to-end-pipeline.md`](./docs/end-to-end-pipeline.md). A contraparte baseada em artifacts de uma execução real está em [`docs/real-run-walkthrough.md`](./docs/real-run-walkthrough.md), usando o run versionado `20260910T115810Z` e diagnósticos reais do `corridor-02`.

`apps/mapping-runtime` compõe workflows de construção de mapa. `apps/map-explorer` abre mapas persistidos para visualização 3D, consulta, inspeção de evidências e exploração de grafo. `apps/cli` fornece acesso scriptável a operações de nível de aplicação.

Agentes de código e contribuidores devem ler [`AGENTS.md`](./AGENTS.md) antes de criar ou alterar código, pastas, interfaces, ou arquitetura de nível de repositório.

## Verificação de desenvolvimento

O módulo executável atual requer Python 3.12. Um ambiente reproduzível para a suíte
completa pode ser preparado e verificado com:

```bash
mise trust .mise.toml  # uma vez, se o ambiente usa mise
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "modules/visual-perception[dev,bench]"
make verify
```

`make verify` executa todos os testes de repositório (inclusive os contracts dos
benchmarks), o lint do repositório inteiro e o `mypy` estrito da API pública de
`visual-perception`. Testes que exigem GPU fazem `skip` explícito quando o hardware ou
o backend opcional não está disponível.

O primeiro slice RGB–LiDAR também possui um smoke test executável sem ROS,
GPU ou dataset externo:

```bash
make m1-test
make m1-demo
make map-explorer-install map-explorer-build
cd apps/map-explorer/web && npm run dev
```

O artifact gerado em `artifacts/m1-demo.json` pode ser aberto diretamente no
seletor do `map-explorer`. O workflow com dados reais permanece dependente da
rosbag, da calibração e da configuração FAST-LIO do sensor usado.

## Documentação

A documentação de arquitetura e integração de nível de repositório está disponível em [`docs/README.md`](./docs/README.md).

Decisões e fluxos importantes são documentados em:

- [`docs/end-to-end-pipeline.md`](./docs/end-to-end-pipeline.md)
- [`docs/real-run-walkthrough.md`](./docs/real-run-walkthrough.md)
- [`docs/architecture.md`](./docs/architecture.md)
- [`docs/system-flow.md`](./docs/system-flow.md)
- [`docs/applications.md`](./docs/applications.md)
- [`docs/map-lifecycle.md`](./docs/map-lifecycle.md)
- [`docs/engineering-principles.md`](./docs/engineering-principles.md)
- [`docs/documentation-policy.md`](./docs/documentation-policy.md)

Documentação detalhada de implementação vive dentro de cada módulo, em `modules/<module>/docs/`, à medida que esses módulos são desenvolvidos.
