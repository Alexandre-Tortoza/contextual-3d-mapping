import assert from "node:assert/strict";
import test from "node:test";

import { compositionEntriesOf, debugGalleryFor, flattenManifestValue, pipelineBackendsOf } from "./debug-explorer-data.js";

// Fixture equivalente a um observations[i] de um context.json schema 3: mistura
// URIs soltas, um sub-objeto de imagens e uma lista de tiles de discovery sob
// a mesma etapa, mais uma segunda etapa com um único documento.
const FRAME_A = {
  observation_id: "obs-a",
  frame_id: "frame-0000",
  debug_manifest: {
    "visual-perception": {
      diagnostics: "vp/diagnostics.json",
      images: {
        raw: "vp/raw.png",
        proposals: "vp/proposals.png",
        "semantic-overlay": "vp/semantic-overlay.png",
      },
      discovery_tiles: [
        { id: "tile-0", input: "vp/tile-0.png", proposals: "vp/tile-0-proposals.json" },
      ],
    },
    "sensor-association": { audit: "sa/audit.json" },
  },
};

// Um segundo frame com uma etapa completamente diferente (grounding em vez de
// discovery_tiles), incluindo uma etapa desconhecida hoje pelo frontend, para
// provar que trocar de frame troca a galeria — e que uma etapa nova aparece
// mesmo sem o código conhecer seu nome.
const FRAME_B = {
  observation_id: "obs-b",
  frame_id: "frame-0007",
  debug_manifest: {
    "visual-perception": {
      diagnostics: "vp/diagnostics-b.json",
      grounding: "vp/grounding-b.json",
      region_views: [
        { region_id: "r0", masked_subject: "vp/r0-masked.png", tight_crop: "vp/r0-tight.png" },
      ],
    },
    "future-stage": { anything: "fs/artifact.png" },
  },
};

const FRAME_EMPTY = { observation_id: "obs-c", frame_id: "frame-0099" };

// Regressão: achatar uma etapa precisa alcançar strings dentro de
// sub-objetos e de listas de objetos, prefixando o caminho para não colidir
// nomes entre etapas diferentes.
test("flattenManifestValue achata objetos e listas aninhadas prefixando o caminho", () => {
  const leaves = flattenManifestValue(FRAME_A.debug_manifest["visual-perception"], "visual-perception");
  const byPath = Object.fromEntries(leaves);
  assert.equal(byPath["visual-perception.diagnostics"], "vp/diagnostics.json");
  assert.equal(byPath["visual-perception.images.raw"], "vp/raw.png");
  assert.equal(byPath["visual-perception.discovery_tiles.0.input"], "vp/tile-0.png");
  assert.equal(byPath["visual-perception.discovery_tiles.0.id"], "tile-0");
});

// A lista de etapas e a separação imagem/documento precisam aparecer para um
// frame concreto, sem presumir nomes fixos de etapa.
test("debugGalleryFor agrupa por etapa e separa imagens de documentos JSON", () => {
  const gallery = debugGalleryFor(FRAME_A);
  assert.deepEqual(gallery.map((group) => group.stage), ["visual-perception", "sensor-association"]);

  const visualPerception = gallery.find((group) => group.stage === "visual-perception");
  const imagePaths = visualPerception.images.map(([path]) => path);
  const documentPaths = visualPerception.documents.map(([path]) => path);
  assert.ok(imagePaths.includes("visual-perception.images.raw"));
  assert.ok(imagePaths.includes("visual-perception.discovery_tiles.0.input"));
  assert.ok(documentPaths.includes("visual-perception.diagnostics"));
  assert.ok(documentPaths.includes("visual-perception.discovery_tiles.0.proposals"));

  const sensorAssociation = gallery.find((group) => group.stage === "sensor-association");
  assert.deepEqual(sensorAssociation.images, []);
  assert.equal(sensorAssociation.documents.length, 1);
});

// É exatamente o que o DebugExplorer faz ao trocar a observação selecionada:
// a galeria derivada muda de etapas e de conteúdo, sem reter nada do frame
// anterior. Etapas desconhecidas (future-stage) continuam aparecendo.
test("trocar de frame troca a galeria exibida", () => {
  const galleryA = debugGalleryFor(FRAME_A);
  const galleryB = debugGalleryFor(FRAME_B);

  assert.deepEqual(galleryA.map((group) => group.stage), ["visual-perception", "sensor-association"]);
  assert.deepEqual(galleryB.map((group) => group.stage), ["visual-perception", "future-stage"]);

  const visualPerceptionA = galleryA.find((group) => group.stage === "visual-perception");
  const visualPerceptionB = galleryB.find((group) => group.stage === "visual-perception");
  assert.notDeepEqual(visualPerceptionA.images, visualPerceptionB.images);

  const futureStage = galleryB.find((group) => group.stage === "future-stage");
  assert.equal(futureStage.images[0][1], "fs/artifact.png");

  assert.deepEqual(debugGalleryFor(FRAME_EMPTY), []);
});

// Uma run isolada publica a composição como uma única string; um mapa
// consolidado publica uma lista com uma entrada por run de origem. As duas
// formas precisam normalizar para a mesma lista nome→uri.
test("compositionEntriesOf normaliza os dois formatos de debug_manifest.composition", () => {
  assert.deepEqual(
    compositionEntriesOf({ debug_manifest: { composition: "composition.json" } }),
    [["composition", "composition.json"]],
  );
  assert.deepEqual(
    compositionEntriesOf({
      debug_manifest: {
        composition: [
          { run_id: "run-a", uri: "runs/run-a/composition.json" },
          { run_id: "run-b", uri: "runs/run-b/composition.json" },
        ],
      },
    }),
    [
      ["run-a", "runs/run-a/composition.json"],
      ["run-b", "runs/run-b/composition.json"],
    ],
  );
  assert.deepEqual(compositionEntriesOf({}), []);
  assert.deepEqual(compositionEntriesOf(null), []);
});

// Mesma lógica de compositionEntriesOf: uma run isolada publica um objeto
// por estágio (sem run_id, já que é a única run), um consolidado publica uma
// lista com um run_id por entrada.
test("pipelineBackendsOf normaliza os dois formatos de debug_manifest.pipeline_backends", () => {
  const singleRunBackends = { region_discovery: { backend: "sam3", checkpoint: "facebook/sam3" } };
  assert.deepEqual(
    pipelineBackendsOf({ debug_manifest: { pipeline_backends: singleRunBackends } }),
    [{ runId: null, backends: singleRunBackends }],
  );
  assert.deepEqual(
    pipelineBackendsOf({
      debug_manifest: {
        pipeline_backends: [
          { run_id: "run-a", backends: { region_discovery: { backend: "sam3" } } },
          { run_id: "run-b", backends: { region_discovery: { backend: "sam2" } } },
        ],
      },
    }),
    [
      { runId: "run-a", backends: { region_discovery: { backend: "sam3" } } },
      { runId: "run-b", backends: { region_discovery: { backend: "sam2" } } },
    ],
  );
  assert.deepEqual(pipelineBackendsOf({}), []);
  assert.deepEqual(pipelineBackendsOf(null), []);
});
