// Agrupa as runs publicadas por uma identidade de comparação explícita. O
// grupo é declarado pelo experimento porque ``map_id`` ou o nome da run não
// provam que os resultados vieram do mesmo frame de entrada.
export function comparisonGroups(entries) {
  const groups = new Map();
  entries.forEach((entry) => {
    if (!entry.comparison || entry.artifactType !== "contextual_rgb_lidar_slice") return;
    const group = groups.get(entry.comparison.groupId) ?? [];
    group.push(entry);
    groups.set(entry.comparison.groupId, group);
  });
  return [...groups.entries()]
    .map(([id, runs]) => ({ id, runs: [...runs].sort((left, right) => left.label.localeCompare(right.label)) }))
    .sort((left, right) => left.id.localeCompare(right.id));
}

// Resolve os eixos que variam dentro do grupo. A página aceita precisamente
// dois deles; expor todos aqui permite informar claramente quando a experiência
// publicou mais dimensões do que uma matriz consegue comunicar sem ambiguidade.
export function matrixAxes(entries) {
  const values = new Map();
  entries.forEach((entry) => {
    Object.entries(entry.comparison?.axes ?? {}).forEach(([axis, option]) => {
      const options = values.get(axis) ?? new Set();
      options.add(option);
      values.set(axis, options);
    });
  });
  return [...values.entries()]
    .filter(([, options]) => options.size > 1)
    .sort(([left], [right]) => {
      const priority = { region_discovery: 0, multimodal_reasoner: 1 };
      return (priority[left] ?? 2) - (priority[right] ?? 2) || left.localeCompare(right);
    })
    .map(([name, options]) => ({ name, options: [...options].sort((left, right) => left.localeCompare(right)) }));
}

// Produto cartesiano de listas de opções por eixo, preservando a atribuição
// eixo->valor de cada combinação. Existe para achatar 1+ eixos de coluna em
// uma única dimensão de grade sem perder qual eixo contribuiu qual valor.
function axisCombinations(axes) {
  return axes.reduce(
    (combinations, axis) => combinations.flatMap((combo) => (
      axis.options.map((option) => [...combo, { axis: axis.name, option }])
    )),
    [[]],
  );
}

// Materializa linhas, colunas e células para a grade. O primeiro eixo (o de
// maior prioridade, ver ``matrixAxes``) vira linha; todo o resto é achatado
// numa única dimensão de coluna, para que 3+ eixos continuem navegáveis numa
// única matriz em vez de exigirem uma tela por combinação de eixos fixados.
// Uma célula ausente é preservada como ``null``: isso torna explícita uma
// combinação que ainda não foi executada, em vez de fazê-la parecer igual a
// uma execução sem regiões.
export function comparisonMatrix(entries) {
  const axes = matrixAxes(entries);
  if (axes.length < 2) return { axes, rows: [], columns: [] };
  const [rowAxis, ...columnAxes] = axes;
  const columnCombinations = axisCombinations(columnAxes);
  const columns = columnCombinations.map((combo) => combo.map((part) => part.option).join(" · "));
  return {
    axes,
    columnAxes,
    columns,
    rows: rowAxis.options.map((row) => ({
      option: row,
      cells: columnCombinations.map((combo) => (
        entries.find((entry) => (
          entry.comparison?.axes[rowAxis.name] === row
          && combo.every((part) => entry.comparison?.axes[part.axis] === part.option)
        )) ?? null
      )),
    })),
  };
}

// Mantém somente observações que existem em todas as runs carregadas. A grade
// troca sempre o mesmo frame, e não mistura por posição listas que poderiam
// ter sido ordenadas diferentemente por cada execução.
export function sharedObservationIds(slices) {
  if (!slices.length) return [];
  const common = new Set((slices[0].observations ?? []).map((item) => item.observation_id));
  slices.slice(1).forEach((slice) => {
    const ids = new Set((slice.observations ?? []).map((item) => item.observation_id));
    [...common].forEach((id) => { if (!ids.has(id)) common.delete(id); });
  });
  return [...common].sort((left, right) => String(left).localeCompare(String(right)));
}
