# Política de Documentação

O repositório usa dois níveis de documentação para manter o contexto arquitetural separado dos detalhes de implementação de módulo.

## Idioma

Toda documentação em prosa — `README.md`, arquivos em `docs/`, docstrings e comentários no código — é escrita em **português do Brasil**.

Permanecem em inglês:

- identificadores de código (nomes de classes, funções, variáveis, módulos, arquivos e diretórios);
- mensagens de commit e títulos/descrições de PR;
- jargão técnico e arquitetural já consagrado (`pipeline`, `backend`, `adapter`, `port`, `framework`, `dataset`, `benchmark`, `checkpoint`, etc.) — usado dentro da prosa em português para não divergir dos nomes usados no código;
- marcadores convencionais como `NOTE:`, `TODO:`, `FIXME:`, `WARNING:` (o texto que os segue vai em português).

Toda função, método e classe é documentada em duas partes: um comentário `#` acima da assinatura explicando o que ela faz, por que existe e, quando relevante, onde é usada; seguido da docstring `"""..."""` traduzida. Ver `AGENTS.md` para o formato completo e exemplo.

## Documentação de nível de repositório

Localização:

```text
docs/
```

Este nível é dono da documentação que se aplica ao projeto como um todo:

- arquitetura do repositório e convenções de diretório;
- topologia de módulos e fluxo de alto nível;
- regras de dependência entre módulos;
- convenções de integração e contract;
- regras de composição de aplicação;
- convenções de teste, avaliação e experimentação de nível de repositório;
- decisões de nível de projeto que afetam mais de um módulo.

A documentação de nível de repositório pode nomear módulos e mostrar como eles se conectam, mas não deve explicar seus algoritmos internos ou escolhas de implementação.

## Documentação de nível de módulo

Localização:

```text
modules/<module>/docs/
```

Cada módulo será dono da sua documentação técnica detalhada quando o desenvolvimento começar. Isso pode incluir:

- objetivos e responsabilidades do módulo;
- contracts públicos de entrada e saída;
- arquitetura interna;
- algoritmos e modelos;
- pré-processamento e pós-processamento;
- workflows de treino e inferência;
- configuração;
- datasets usados especificamente pelo módulo;
- benchmarks e metodologia de avaliação;
- justificativa de implementação;
- limitações e modos de falha conhecidos;
- referências de pesquisa relevantes para essa implementação.

Cada módulo deve eventualmente expor um ponto de entrada como:

```text
modules/<module>/docs/index.md
```

## Regra de separação

Se um documento responde **como o sistema é composto**, ele pertence à raiz `docs/`.

Se um documento responde **como um módulo funciona**, ele pertence ao `docs/` daquele módulo.

A documentação raiz deve linkar para a documentação de módulo quando necessário, em vez de duplicá-la. Isso evita que implementações de pesquisa em evolução independente tornem a documentação de arquitetura global instável.
