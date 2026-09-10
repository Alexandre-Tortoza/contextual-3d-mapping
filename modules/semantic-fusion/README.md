# Semantic Fusion

`semantic-fusion` decide qual classificação um ponto do mapa persistente carrega quando
várias observações visuais o classificam, e preserva todas as propostas concorrentes.

## Responsabilidades

- receber contribuições semânticas ancoradas na mesma identidade de geometria;
- escolher um label primário por uma regra explícita e determinística;
- medir a concordância entre as observações contribuintes;
- preservar cada contribuinte com sua observação, região e confiança de origem.

## Não-responsabilidades

- projeção, visibilidade e oclusão (`sensor-association`);
- persistência das entidades semânticas resultantes (`semantic-map`);
- calibração da confiança dos claims (`visual-perception`).

## Por que o módulo existe

Enquanto um trecho era contextualizado por um único keyframe, nenhum ponto recebia duas
classificações e a fusão não tinha o que decidir. Com múltiplos keyframes cobrindo o
mesmo trecho, a mesma superfície é observada de vários ângulos e distâncias, e as
propostas divergem. Sem uma regra explícita, a última gravação venceria por acidente de
ordem de execução — o que tornaria o mapa dependente da ordem em que os frames foram
processados.

## Regra de fusão

`fuse_point_contributions` ordena as contribuições por:

1. confiança calibrada, quando o produtor a fornece;
2. confiança bruta declarada, como alternativa;
3. fração de evidência visual favorável (`visual_support`);
4. qualidade geométrica da máscara (`region_quality`);
5. instante da observação e identificador da região, como desempate determinístico.

Os sinais de suporte entram apenas como desempate, e não como soma ponderada, porque uma
combinação arbitrária deles fingiria uma calibração que o pipeline visual ainda não tem
(veja as issues de calibração semântica). Quando a calibração existir, ela passa a
dominar a ordenação sem que a regra mude.

`agreement` é a fração de contribuições que concordam com o label primário. Ela é
evidência: um label sustentado por cinco observações não tem o mesmo peso de um
sustentado por uma.

## Fronteira pública

```text
SemanticContribution   proposta de uma observação para um ponto
FusedPointContext      label primário, concordância e contribuintes preservados
fuse_point_contributions
```

Nenhuma contribuição é descartada pela fusão. O label primário é uma escolha, não um
apagamento das alternativas, e o artifact preserva as duas coisas para inspeção.
