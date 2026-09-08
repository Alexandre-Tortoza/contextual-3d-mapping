# Geometria de área por sequência

Cada arquivo aqui declara, para uma sequência, as duas áreas de imagem que
limitam a evidência que `visual-perception` pode usar:

- `valid_area` — a parte útil do sensor. Em `corridor-02` é o círculo da lente
  fisheye; fora dele os pixels são o corpo preto da objetiva, não cena;
- `ego_vehicle` — a área ocupada pelo próprio rig, que aparece fixa em todo
  frame da sequência.

## Por que geometria declarada, e não derivada em runtime

Os intrínsecos de `corridor-02`
(`datasets/raw/corridor-02/corridor-02-Intrinsics.yaml`) usam o modelo MEI com
ξ = 1,563. Nem o limite de campo de visão (θ ≈ 130°) nem o de desprojeção
(ρ ≤ 0,83, cerca de 565 px) delimitam o anel preto: os dois excedem a diagonal
da imagem. **A vinheta é óptica, não geométrica** — não há como derivá-la da
calibração disponível.

A alternativa determinística é medir uma vez, offline, sobre a sequência
inteira, e versionar o resultado. A pipeline lê a geometria declarada; ela
nunca infere área por limiar de cor a cada frame.

Aplicar uma área **não altera pixel nenhum**. A exclusão acontece na filtragem
de proposals, sobre máscaras — ver `docs/known-limitations.md` para o custo
medido de fazer diferente.

## Regenerar

```bash
python benchmarks/derive_sequence_masks.py --sequence-id corridor-02
```

O script reajusta `valid_area` a partir dos frames locais, imprime o erro de
ajuste e **preserva** `ego_vehicle`, que é traçado à mão sobre o frame e
revisado visualmente.
