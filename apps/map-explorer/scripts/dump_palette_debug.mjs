#!/usr/bin/env node

/**
 * Dump de paleta e famílias de um artifact já aberto, para auditar o que
 * o viewer construiria sem precisar abrir o browser.
 *
 * Uso: node dump_palette_debug.mjs <artifact.json> [output.json]
 */

import fs from "fs";
import path from "path";

// Cópia da lógica de STRUCTURAL_HEAD_NOUNS de map-data.js
const STRUCTURAL_HEAD_NOUNS = Object.freeze({
  carpet: true,
  ceiling: true,
  floor: true,
  flooring: true,
  grass: true,
  ground: true,
  pavement: true,
  road: true,
  sky: true,
  surface: true,
  tile: true,
  wall: true,
  baseboard: true,
  molding: true,
  panel: true,
  panelling: true,
  partition: true,
  plank: true,
});

const UNOBSERVED_KEY = "__unobserved__";
const OBSERVED_UNLABELED_KEY = "__observed_unlabeled__";
const STRUCTURAL_KEY = "__structural__";
const OTHER_FAMILY_KEY = "__other__";
const FAMILY_COLORS = [
  [57, 135, 229],
  [201, 133, 0],
  [213, 81, 129],
  [0, 131, 0],
];

// Converte sRGB para linear (match map-data.js).
function srgbColorToLinear(color) {
  return color.map((channel) => {
    const normalized = channel / 255;
    return normalized <= 0.04045
      ? normalized / 12.92
      : Math.pow((normalized + 0.055) / 1.055, 2.4);
  });
}

// Função auxiliar: pega evidência do ponto.
function visualEvidence(point) {
  return point.association ?? point.context ?? null;
}

// Função auxiliar: checa se ponto foi visualmente observado.
function isVisuallyObserved(evidence) {
  if (!evidence) return false;
  if (evidence.status) return evidence.status === "associated";
  return Boolean(evidence.color_rgb || evidence.pixel);
}

// Função auxiliar: checa se label é estrutural.
function isStructuralLabel(label) {
  if (!label) return false;
  const tokens = label.trim().toLowerCase().split(/\s+/);
  if (tokens.length === 0) return false;
  const headNoun = tokens[tokens.length - 1];
  return headNoun in STRUCTURAL_HEAD_NOUNS;
}

// Função auxiliar: classifica cada ponto em uma categoria.
function contextKey(point) {
  const evidence = visualEvidence(point);
  if (evidence?.label) {
    const label = evidence.label.trim().toLowerCase();
    if (isStructuralLabel(label)) return STRUCTURAL_KEY;
    return label;
  }
  if (isVisuallyObserved(evidence)) return OBSERVED_UNLABELED_KEY;
  return UNOBSERVED_KEY;
}

// Verifica se uma sequência de tokens aparece inteira dentro de outra.
function containsTokens(tokens, needle) {
  if (needle.length >= tokens.length) return false;
  for (let start = 0; start + needle.length <= tokens.length; start += 1) {
    if (needle.every((token, offset) => tokens[start + offset] === token))
      return true;
  }
  return false;
}

// Agrupa labels por família (merge textual).
function buildLabelFamilies(labelCounts) {
  const ordered = [...labelCounts.entries()].sort(
    (left, right) => right[1] - left[1] || left[0].localeCompare(right[0])
  );
  const host = new Map();
  ordered.forEach(([label]) => {
    const tokens = label.split(/\s+/);
    const parent = ordered.find(
      ([candidate]) =>
        candidate !== label && containsTokens(tokens, candidate.split(/\s+/))
    );
    host.set(label, parent ? parent[0] : label);
  });
  const families = new Map();
  host.forEach((_, label) => {
    const seen = new Set();
    let root = label;
    while (host.get(root) !== root && !seen.has(root)) {
      seen.add(root);
      root = host.get(root);
    }
    families.set(label, root);
  });
  return families;
}

// Constrói paleta (versão simplificada, só o mapeamento).
function buildContextPalette(points) {
  const labelCounts = new Map();
  const neutralCounts = new Map();
  const structuralCounts = new Map();

  points.forEach((point) => {
    const evidence = visualEvidence(point);
    if (evidence?.label) {
      const label = evidence.label.trim().toLowerCase();
      if (isStructuralLabel(label)) {
        structuralCounts.set(label, (structuralCounts.get(label) ?? 0) + 1);
      } else {
        labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
      }
    } else {
      const key = contextKey(point);
      neutralCounts.set(key, (neutralCounts.get(key) ?? 0) + 1);
    }
  });

  const families = buildLabelFamilies(labelCounts);
  const grouped = new Map();
  labelCounts.forEach((count, label) => {
    const family = families.get(label) ?? label;
    const entry = grouped.get(family) ?? { key: family, count: 0, members: [] };
    entry.count += count;
    entry.members.push({ key: label, label, count });
    grouped.set(family, entry);
  });

  const ranked = [...grouped.values()].sort(
    (left, right) => right.count - left.count || left.key.localeCompare(right.key)
  );

  const colorByFamily = new Map();
  ranked.forEach((entry, rank) => {
    const color = FAMILY_COLORS[rank] || [185, 189, 199]; // OTHER_FAMILY_COLOR
    colorByFamily.set(entry.key, color);
  });

  return {
    legend: {
      structural: structuralCounts,
      interesting: labelCounts,
      neutrals: neutralCounts,
    },
    colorByFamily,
    families,
    ranked,
  };
}

// Main
const args = process.argv.slice(2);
if (args.length < 1) {
  console.error(
    "Uso: node dump_palette_debug.mjs <artifact.json> [output.json]"
  );
  process.exit(1);
}

const artifactPath = args[0];
const outputPath = args[1] || path.join(
  path.dirname(artifactPath),
  "DEBUG",
  "viewer.json"
);

try {
  const artifactData = JSON.parse(fs.readFileSync(artifactPath, "utf-8"));
  const points = artifactData.points || [];

  if (!points.length) {
    console.warn("Nenhum ponto no artifact.");
    process.exit(0);
  }

  const palette = buildContextPalette(points);

  // Prepara saída.
  const output = {
    artifact: path.basename(artifactPath),
    total_points: points.length,
    structural_labels: {
      count: [...palette.legend.structural.values()].reduce((a, b) => a + b, 0),
      breakdown: Object.fromEntries(palette.legend.structural),
    },
    interesting_families: palette.ranked.slice(0, FAMILY_COLORS.length).map((entry) => ({
      family: entry.key,
      count: entry.count,
      color: palette.colorByFamily.get(entry.key),
      members: entry.members.slice(0, 5), // amostra
    })),
    tail_bucket: {
      count: palette.ranked
        .slice(FAMILY_COLORS.length)
        .reduce((sum, e) => sum + e.count, 0),
      families_count: palette.ranked.length - FAMILY_COLORS.length,
    },
    neutral_coverage: {
      unobserved: palette.legend.neutrals.get(UNOBSERVED_KEY) || 0,
      observed_unlabeled: palette.legend.neutrals.get(OBSERVED_UNLABELED_KEY) || 0,
    },
  };

  // Cria diretório se necessário.
  const outputDir = path.dirname(outputPath);
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  fs.writeFileSync(outputPath, JSON.stringify(output, null, 2), "utf-8");
  console.log(`✓ Debug dumped to ${outputPath}`);
} catch (error) {
  console.error(`Erro ao processar artifact: ${error.message}`);
  process.exit(1);
}
