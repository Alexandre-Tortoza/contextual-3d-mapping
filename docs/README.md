# Documentação

Este diretório contém a documentação de nível de repositório do `contextual-3d-mapping`.

A documentação aqui descreve o projeto como um sistema: sua intenção arquitetural, fronteiras de módulo, princípios de engenharia, regras de integração, composição de aplicação, fronteiras de persistência e fluxo de ponta a ponta.

O projeto usa intencionalmente uma arquitetura modular simples orientada a capacidades, em vez de Clean Architecture como padrão de nível de repositório.

A documentação detalhada de implementação vive ao lado de cada módulo, em:

```text
modules/<module>/docs/
```

Essa documentação local pode descrever algoritmos, escolhas de modelo, procedimentos de treino ou inferência, configuração, arquitetura interna, benchmarks, decisões de implementação, limitações e referências de pesquisa específicas do módulo.

## Documentação do repositório

- [Arquitetura](./architecture.md): arquitetura de nível de repositório, ownership de capacidades, fronteiras e regras de dependência.
- [Princípios de engenharia](./engineering-principles.md): decisões de implementação, interpretação de SOLID, limiares de abstração, localidade de integração, testes e diretrizes de review.
- [Fluxo do sistema](./system-flow.md): fluxo entre módulos e entre aplicações.
- [Aplicações](./applications.md): responsabilidades e fronteiras das aplicações executáveis.
- [Ciclo de vida do mapa](./map-lifecycle.md): ciclo de vida desde observações até mapas persistidos e consultáveis.
- [Política de documentação](./documentation-policy.md): separação entre documentação global e documentação local de módulo.
- [`AGENTS.md`](../AGENTS.md): regras condensadas do repositório que agentes de código e contribuidores devem ler antes de alterar arquitetura ou código.

## Resumo da arquitetura

O repositório segue um pequeno conjunto de regras estáveis:

1. cada capacidade tem um dono de módulo óbvio;
2. módulos expõem APIs públicas pequenas e escondem detalhes de implementação;
3. aplicações compõem capacidades em vez de possuir algoritmos de pesquisa;
4. integrações e contracts específicos de capacidade ficam próximos dos módulos que os possuem;
5. código compartilhado é limitado a primitivas genuinamente de nível de repositório e estáveis;
6. abstrações são introduzidas para fronteiras concretas ou pontos de variação, não especulativamente;
7. SOLID guia o design de código e dependências, não a nomenclatura de pastas;
8. papers, modelos, repositórios e datasets externos podem influenciar implementações sem definir a arquitetura pública.

## Fronteira da documentação

O `docs/` raiz responde perguntas como:

- Por que o repositório está organizado desta forma?
- Como os módulos são compostos?
- Qual capacidade possui uma determinada responsabilidade?
- Quais direções de dependência são permitidas?
- Quando um contract ou abstração compartilhada deve existir?
- Como a informação se move pelo sistema em alto nível?
- Qual aplicação constrói mapas e qual aplicação os explora?
- Onde a persistência se encaixa arquiteturalmente?
- Quais regras de engenharia devem permanecer estáveis conforme as implementações de pesquisa mudam?

Ele não deve explicar o algoritmo interno de um módulo específico. Esses detalhes pertencem ao `docs/` daquele módulo.
