# Política de anotação do conjunto de referência

Issue: #197. Schema: `visual-reference/1`
(`datasets/contextual_mapping_datasets/annotation_manifest.py`).
Manifest: `datasets/manifests/corridor-02-visual-reference.json`.
Partição fixa: `datasets/splits/corridor-02-visual-reference-1.json`.

Este documento define o que anotar, o que ignorar e como resolver
divergência no conjunto de referência usado para avaliar percepção visual.
Ele é a versão `annotation-policy/1` referenciada por
`AnnotationProvenance.policy_version` em cada amostra.

## Por que uma política, e não só um formato

O formato do manifest impede erros estruturais — identidade duplicada,
referência pendurada, vazamento de split. Ele não consegue impedir o erro
que mais compromete um resultado de pesquisa: anotar por suposição. Um
anotador que precisa preencher um label acaba inventando um. Esta política
existe para dar a ele as saídas explícitas — `ambiguous`, `unknown`,
`ignored` — e para tornar o uso delas a resposta certa, e não um sinal de
trabalho incompleto.

## Escopo do conjunto

O conjunto cobre a sequência `corridor-02` (ambiente interno, câmera
fisheye montada em um robô terrestre). Os frames são extraídos por
`experiments/visual_perception_experiments/prepare_reference.py`, que é
determinístico: a mesma sequência sempre produz as mesmas amostras.

### Inclusão

Uma amostra entra no conjunto quando:

- foi selecionada pela amostragem temporal uniforme dentro do seu bloco;
- tem resolução completa e decodificou sem erro;
- não está inteiramente saturada (branco/preto) a ponto de nenhuma região
  ser distinguível.

### Exclusão

Uma amostra é removida do manifest (e não apenas marcada) quando ela não
contém nenhum conteúdo interpretável — por exemplo, um frame totalmente
escuro durante uma transição de exposição. A remoção é registrada no commit
que altera o manifest.

### Pré-processamento aplicado

A faixa inferior do frame (a partir de `y = 340` em 640x480) é zerada antes
da gravação: ela contém o chassi e as rodas do próprio robô, que aparecem
identicamente em todos os frames e não fazem parte do ambiente mapeado.
A amostra registra isso na condição de captura `ego_vehicle_masked`.
Anotador e modelo enxergam exatamente os mesmos pixels.

## Splits

A sequência é dividida em três blocos temporais contíguos —
`development` (40%), `calibration` (25%), `test` (35%) — separados por
*guard bands* de 3% que são descartadas.

As bandas existem porque frames vizinhos de um vídeo são quase idênticos.
Sem elas, o último frame de `development` e o primeiro de `calibration`
mostrariam a mesma parede, e o resultado de teste estaria contaminado antes
da primeira métrica. A validação do manifest recusa a mesma imagem (por
digest) ou o mesmo frame de origem em mais de um split.

## O que anotar

### Geometria de região

Anote como região qualquer superfície ou objeto que o mapeamento precise
distinguir: paredes, piso, teto, portas, batentes, extintores, placas,
caixas, corrimãos, aberturas. Inclua deliberadamente:

- **estruturas pequenas** (maçanetas, interruptores, sinalizações);
- **estruturas finas** (corrimãos, batentes, cabos, pés de móvel);
- **regiões parcialmente ocluídas**, anotando apenas a parte visível.

A máscara acompanha o contorno visível do objeto, não a sua extensão
inferida atrás de uma oclusão.

### Semântica de vocabulário aberto

`labels` é uma lista de termos aceitáveis, não uma classe. Escreva os
termos em português ou inglês conforme o uso corrente da equipe, em
minúsculas, sem artigo.

- **`certain`**: há um termo claramente correto. Liste-o; liste sinônimos
  que você aceitaria de um sistema (`porta`, `door`).
- **`ambiguous`**: dois ou mais termos são defensáveis olhando o frame
  (`porta` ou `armário`). Liste todos. Uma predição que acerte qualquer um
  deles conta como correta — é exatamente por isso que listar todos importa.
- **`unknown`**: a região existe e é visualmente distinta, mas você não
  consegue nomeá-la com confiança. Deixe `labels` vazio. Ela continua
  contando para a métrica de detecção e sai da métrica de semântica.

Não existe taxonomia fechada. Não force um termo por não haver opção na
lista de outra amostra.

### Atributos, condição, material e hazard

Preencha apenas quando forem visualmente evidentes no frame:

- `attributes`: estado observável (`aberta`, `fechada`, `molhada`);
- `condition`: integridade (`íntegra`, `danificada`, `obstruída`);
- `material`: material aparente (`madeira`, `metal`, `concreto`);
- `hazards`: risco visível (`piso molhado`, `obstrução de saída`).

Deixar em branco é a resposta correta quando não é evidente. Uma tarefa sem
anotação fica *sem suporte* na avaliação, e nunca é contada como zero.

### Relações

Anote relações apenas entre regiões anotadas na mesma amostra, com
predicados em snake_case (`near`, `contains`, `overlaps`, `supports`).
Anote a relação apenas quando ela for verificável no próprio frame 2D.

### Visibilidade

Preencha `visibility` com um termo curto (`clear`, `partial`, `degraded`)
quando for relevante para estratificar o resultado. O campo alimenta o
recorte por estrato do relatório.

## O que ignorar

Marque `ignored = true` (e deixe `labels` vazio) quando a região existe mas
não deve influenciar a métrica:

- conteúdo cortado pela borda do frame, sem extensão suficiente para julgar;
- regiões severamente borradas por movimento;
- regiões saturadas por reflexo ou sombra a ponto de o conteúdo ser
  indecidível;
- a faixa do chassi do robô, caso alguma parte escape do mascaramento.

Uma região ignorada **absorve** a predição que a cobre: essa predição não
conta como acerto nem como falso positivo. É o que impede o conjunto de
punir um sistema por acertar algo que nós mesmos não sabemos julgar.

`ignored` nunca é usado para descartar uma região difícil que você
*consegue* julgar. Para essa, use `ambiguous` ou `unknown`.

## Controle de qualidade

1. **Dupla anotação** das amostras de `test`. Cada amostra recebe duas
   passagens independentes.
2. **Medida de concordância** por IoU de máscara (limiar 0,5) e por
   sobreposição de conjunto de labels.
3. **Resolução de divergência**: um terceiro revisor decide. Quando os dois
   labels forem defensáveis, a resolução correta é `ambiguous` com os dois
   termos, não a escolha de um deles. A decisão é registrada em
   `AnnotationProvenance.resolution`.
4. **Revisão final**: a amostra passa a `reviewed`. Um conjunto com qualquer
   amostra fora de `reviewed` é recusado por
   `validate_reference(..., require_reviewed=True)`, que é a checagem que
   antecede a publicação de qualquer número.

## Ferramentas

```bash
# Extrai os frames e cria o manifest pending-review (determinístico).
python -m visual_perception_experiments.prepare_reference --frames-per-split 12

# Gera o pacote HTML autocontido de revisão.
python -m visual_perception_experiments.review_package

# Opcional: transforma predições reais em um manifest de rascunho separado.
python -m visual_perception_experiments.draft_reference \
  --predictions <predictions.json> \
  --out <reference-draft.json>
```

O pacote de revisão mostra cada frame com overlays das masks/labels, sua identidade, seu
digest, suas condições de captura medidas e o que já foi anotado, junto desta checklist.
O gerador de rascunho sempre grava `model_assisted_draft` e
`pending_review`; ele não sobrescreve o manifest canônico nem pode satisfazer
`require_reviewed=True`. O revisor humano precisa corrigir masks, labels, ambiguidade e
`ignored`, e o split `test` continua sujeito às duas passagens independentes descritas
acima.

## Estado atual

O manifest versionado contém 36 amostras (12 por split), todas em
`pending_review` e sem regiões anotadas. Nenhum número de qualidade
semântica publicado por este repositório pode se apoiar nele até que a
revisão humana aconteça; os experimentos que dependem de anotação declaram
essa limitação explicitamente em seus relatórios.
