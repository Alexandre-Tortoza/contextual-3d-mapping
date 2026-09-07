# Handoff: concluir a percepção contextual e repetir a validação em três frames

## Objetivo do próximo agente

Concluir as capacidades ainda abertas que afetam a qualidade visual e semântica da
pipeline canônica, executar novamente os mesmos três frames com os backends reais e
comparar os novos overlays e artifacts com as três imagens que o usuário enviou no
início do trabalho.

Não declarar que o resultado ficou melhor apenas porque diminuiu o número de regiões ou
porque o audit estrutural passou. A comparação visual deve ser explícita, frame a frame,
e qualquer conclusão quantitativa de qualidade semântica depende de ground truth humano
revisado.

## Estado recebido

- Data do handoff: 2026-09-06.
- Branch: `main`.
- Revisão no momento deste handoff: `667de9f`.
- `main` estava limpa e sincronizada com `origin/main` antes da criação deste arquivo.
- As issues `#208`, `#209`, `#212` e `#213` estão fechadas.
- A suíte reproduzível está exposta por `make verify`.
- A validação real aceita seleção determinística por `--frame-id`, registra os hashes
  das entradas e separa a saída canônica de pós-processamento opcional.
- O perfil `full` produziu os quatro slots de evidência para 126/126 regiões nos três
  frames, sem falhas de evidência, interpretação ou calibração.
- O pico observado nas execuções recentes foi 4,57 GiB, dentro do budget de 8 GiB da
  RTX 3060.

Commits recentes relevantes:

- `9ae3a4d`: workflows reproduzíveis, elevação real de features, validação selecionável
  e saúde da suíte;
- `0bc03b2`: complementos de relatório do benchmark e auditoria de splits;
- `ebe211f`: benchmark medido de elevação de features;
- `667de9f`: artifacts reais da ablation multi-contexto em três frames.

## Os três frames vinculantes

Use exatamente estes IDs, nesta ordem:

1. `corridor-02-000`
2. `corridor-02-008`
3. `corridor-02-017`

Hashes esperados dos PNGs de entrada:

| Frame | SHA-256 |
| --- | --- |
| `corridor-02-000` | `4a69f4d01bed3534b7581aa6199a45fb3be7b91a405a58bc44af29261f3cf1e6` |
| `corridor-02-008` | `2a0db0d9ee1ad0e3d3f05785c028b0c63221194e7589c55c6b0ad1c73601b78b` |
| `corridor-02-017` | `f7ea622db269c79244af5dfc5ae66bd5d1fbe2b5babee60cb51d71acccd5a20a` |

Os arquivos locais esperados ficam em:

```text
benchmarks/.local/corridor-02-frames/
```

Se estiverem ausentes, consulte `benchmarks/prepare_corridor02_frames.py`; não substitua
os frames por outros, pois isso invalidaria a comparação.

## Baselines que não devem ser apagadas

### Resultado histórico anterior às melhorias recentes

Use como baseline persistida no repositório, caso as três imagens originais da conversa
correspondam a estes frames:

```text
benchmarks/results/samples/20260904T112627Z/
```

Os overlays comparáveis são:

```text
benchmarks/results/samples/20260904T112627Z/corridor-02-000.overlay.png
benchmarks/results/samples/20260904T112627Z/corridor-02-008.overlay.png
benchmarks/results/samples/20260904T112627Z/corridor-02-017.overlay.png
```

Essa execução foi produzida na revisão `36ef954` e reportou 91, 60 e 87 regiões,
respectivamente. Contagem diferente não implica, sozinha, melhora ou regressão.

### Checkpoint atual antes do próximo ciclo

O run `full` mais recente é:

```text
benchmarks/results/samples/20260906T195310Z/
```

Ele foi produzido em `ebe211f` e reportou 54, 22 e 50 regiões canônicas, zero falhas de
interpretação e audit aprovado nos três frames. Seus overlays ficam em `canonical/`.

O run comparável com contexto reduzido é:

```text
benchmarks/results/samples/20260906T200203Z/
```

O relatório consolidado da comparação baseline versus `full` está em:

```text
benchmarks/results/benchmark-209-multi-context-20260906T201128Z.md
```

O benchmark de resolução realmente elevada está em:

```text
benchmarks/results/benchmark-208-feature-elevation-20260906T195206Z.md
```

### Limite importante sobre as três imagens da conversa

Imagens anexadas à conversa não possuem necessariamente um caminho persistente no
workspace. Antes de concluir a comparação, confirme visualmente que as imagens originais
do usuário correspondem aos três overlays históricos acima. Se o próximo agente não
tiver acesso aos anexos, deve pedir que o usuário os reenvie; não deve assumir que são
idênticos apenas pelos nomes dos frames.

## Trabalho restante recomendado

Priorize mudanças que possam alterar o resultado observado nos três frames:

1. Auditar a issue `#201` contra a implementação já entregue em `#208`. Fechar `#201`
   somente se cache, serialização, fallback e casos full-frame/resized/tiled/cropped
   satisfizerem integralmente seus critérios; caso contrário, implementar a lacuna.
2. Implementar `#202`: preservar contexto de cena estruturado para o raciocínio de
   região, sem reduzi-lo a uma string.
3. Implementar `#203`: fazer a semântica de região consumir foreground mask-aware,
   tight crop, contextual crop e contexto de cena como evidências distinguíveis.
4. Implementar `#204`: dirigir refinamento por estado de suporte/evidência e registrar
   razão, caminho e histórico append-only.
5. Implementar `#205`: reconciliação contextual intra-frame sem alterar geometria nem
   apagar hipóteses anteriores.
6. Implementar `#206`: relações semânticas estruturadas entre IDs canônicos, mantendo-as
   separadas das relações geométricas.
7. Concluir `#207`: integrar esses estágios em uma única ordem determinística dentro de
   `run_canonical_pipeline`, com audit final após os estágios que adicionam claims ou
   relações.
8. Completar em `#190` o que ainda faltar para provar round-trip canônico, segunda
   execução com reutilização válida de cache e registro do ambiente real.

Respeite o ownership do módulo e mantenha backends nos adapters, comportamento de
capacidade em `application/`/`domain/`, configuração algorítmica no módulo e composição
explícita no entrypoint.

## Bloqueio de avaliação quantitativa

As issues `#210` e `#211` continuam abertas:

- `#210` exige anotação humana revisada, inclusive dupla anotação do split `test`;
- `#211` ajusta e valida a calibração somente depois de `#210`.

O manifest atual possui 36 amostras, todas `pending_review` e sem regiões humanas
revisadas. O próximo agente pode preparar o pacote de revisão e rascunhos assistidos,
mas não pode promover predições do modelo a ground truth nem marcar a revisão humana
como concluída.

Enquanto esse bloqueio existir, `#199` e `#200` podem avançar em infraestrutura e
relatórios operacionais, mas não devem publicar ECE, Brier, taxa de hallucination ou
superioridade semântica como resultado do dataset real.

## Verificação antes da execução real

A partir da raiz do repositório:

```bash
git status --short --branch
nvidia-smi
make verify PYTHON=modules/visual-perception/.venv/bin/python
```

O esperado é:

- worktree limpa antes da implementação;
- RTX 3060 visível;
- testes, `ruff` e `mypy` aprovados;
- testes de GPU executados quando as dependências `ml` estiverem instaladas, ou `skip`
  explicitamente justificado quando não estiverem.

Não execute a pipeline real a partir de código não commitado. O `manifest.json` deve
apontar para uma revisão Git que contenha exatamente o comportamento avaliado.

## Comando da nova execução de três frames

A partir de `modules/visual-perception/`, com `.venv` contendo os extras reais:

```bash
./.venv/bin/python benchmarks/validate_reference_pipeline.py \
  --context-profile full \
  --frame-id corridor-02-000 \
  --frame-id corridor-02-008 \
  --frame-id corridor-02-017
```

Use `--semantic-merge` somente para gerar uma variante adicional. A comparação principal
deve usar `canonical/`, pois esse diretório preserva exatamente a saída de
`run_canonical_pipeline`. Nunca substitua os artifacts canônicos pelos pós-processados.

O novo run será gravado em:

```text
benchmarks/results/samples/<novo-run-id>/
```

Antes de analisar, confirme no novo `manifest.json`:

- `git_revision` igual ao commit testado;
- `ordered_frame_ids` exatamente na ordem vinculante;
- os três SHA-256 iguais aos registrados neste handoff;
- `context_profile` igual a `full`;
- `config_fingerprint` presente;
- ausência de OOM e fallback silencioso;
- audit, falhas, slots, latência, chamadas e VRAM presentes por frame.

## Como decidir se os resultados melhoraram

Abra lado a lado, para cada frame:

1. a imagem/overlay original enviada pelo usuário;
2. o overlay histórico de `20260904T112627Z`;
3. o checkpoint `full` de `20260906T195310Z/canonical/`;
4. o novo overlay em `<novo-run-id>/canonical/`.

Avalie e registre separadamente:

| Eixo | Pergunta |
| --- | --- |
| geometria | masks aderem melhor aos objetos e evitam fundo, buracos e fragmentação? |
| regiões pequenas/finas | portas, bordas, pilares e estruturas distantes continuam representadas? |
| over-segmentation | superfícies repetitivas deixam de gerar regiões redundantes sem apagar objetos reais? |
| labels | o label descreve o foreground da mask, em vez do contexto global ou do crop inteiro? |
| incerteza | casos ambíguos ficam sem score, abstidos ou com alternativas, em vez de certeza inventada? |
| contexto | scene context ajuda sem promover uma propriedade global a verdade da região? |
| relações | relações semânticas usam IDs válidos, são plausíveis e permanecem distintas das geométricas? |
| robustez | não há falhas locais ocultas, artifacts ausentes ou mudança de geometria causada por reasoning? |
| custo | latência, chamadas e VRAM permanecem justificáveis frente ao ganho observado? |

Para cada frame, produza uma conclusão `melhor`, `misto`, `equivalente` ou `pior`, com
dois ou três exemplos visuais concretos. Depois dê uma conclusão global. Não use apenas
a contagem de warnings: muitos warnings podem ser informativos, e audit aprovado mede
validade estrutural, não correção semântica.

Se houver ground truth humano revisado até lá, rode também a avaliação versionada e
fundamente a conclusão em métricas por estrato. Sem ground truth, rotule a conclusão
como **comparação qualitativa de três frames**, não como benchmark de acurácia.

## Critérios de encerramento do próximo ciclo

- implementação e documentação commitadas em uma revisão identificável;
- `make verify` aprovado;
- novo run real com os três hashes vinculantes;
- artifacts canônicos preservados e versionados;
- comparação visual frame a frame entregue ao usuário;
- limitações e resultados negativos registrados;
- somente issues com todos os critérios comprovados fechadas;
- worktree limpa e `main` sincronizada ao final, se o usuário mantiver essa autorização.
