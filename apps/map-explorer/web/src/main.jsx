import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls, PointMaterial } from "@react-three/drei";
import * as THREE from "three";
import "./styles.css";

const COLOR_MODES = { geometry: "Geometria", rgb: "RGB", context: "Contexto" };

// Escolhe a cor sem misturar altura, cor física e claim semântico. Pontos sem
// evidência permanecem neutros nas camadas que dependem de observação visual.
function pointColor(point, colorMode) {
  if (colorMode === "context") return point.association?.semantic_color_rgb ?? [48, 53, 64];
  if (colorMode === "rgb") return point.association?.color_rgb ?? [48, 53, 64];
  return point.display_color_rgb ?? [120, 130, 145];
}

// Converte o artifact público em buffers de renderização; o cliente nunca lê
// detalhes do backend de geometria ou associação.
function Cloud({ points, colorMode, onSelect }) {
  const geometry = useMemo(() => {
    const positions = [];
    const colors = [];
    points.forEach((point) => {
      positions.push(...point.coordinates_m);
      colors.push(...pointColor(point, colorMode).map((channel) => channel / 255));
    });
    const value = new THREE.BufferGeometry();
    value.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    value.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return value;
  }, [points, colorMode]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points geometry={geometry} onClick={(event) => onSelect(points[event.index] ?? null)}>
      <PointMaterial vertexColors size={2} sizeAttenuation={false} />
    </points>
  );
}

// Marca o ponto inspecionado sem alterar Bounds nem reposicionar a câmera.
// Existe para manter seleção e navegação como estados independentes.
function SelectionMarker({ point }) {
  if (!point) return null;
  return (
    <mesh position={point.coordinates_m}>
      <sphereGeometry args={[0.08, 16, 16]} />
      <meshBasicMaterial color="#ff3b30" depthTest={false} />
    </mesh>
  );
}

// Valida somente a fronteira que o viewer consome, produzindo um diagnóstico
// legível quando o usuário seleciona outro JSON por engano.
function validateSlice(value) {
  if (![1, 2].includes(value?.schema_version)) {
    throw new Error("O artifact precisa usar schema_version 1 ou 2.");
  }
  if (typeof value.map_id !== "string" || typeof value.map_frame !== "string") {
    throw new Error("O artifact precisa declarar map_id e map_frame.");
  }
  if (!Array.isArray(value.points)) throw new Error("O artifact precisa conter uma lista points.");
  value.points.forEach((point, index) => {
    if (!Array.isArray(point.coordinates_m) || point.coordinates_m.length !== 3) {
      throw new Error(`O ponto ${index} não possui coordinates_m tridimensional.`);
    }
  });
  if (value.schema_version === 2 && (!Array.isArray(value.observations) || !Array.isArray(value.regions))) {
    throw new Error("O artifact contextual precisa declarar observations e regions.");
  }
  return value;
}

// Explica o alcance real do artifact sem transformar cobertura parcial em uma
// promessa de mapa semântico completo.
function Coverage({ slice }) {
  if (!slice.context_summary) return <p className="notice">Este artifact contém somente geometria.</p>;
  const summary = slice.context_summary;
  return (
    <p className="notice">
      <strong>{summary.contextual_point_count.toLocaleString("pt-BR")}</strong> pontos com contexto e{" "}
      <strong>{summary.rgb_point_count.toLocaleString("pt-BR")}</strong> com RGB;{" "}
      {summary.unobserved_geometric_point_count.toLocaleString("pt-BR")} pontos geométricos ainda não observados.
      Claims: {summary.claim_status}.
    </p>
  );
}

// Resolve um asset relativo ao JSON servido. Upload local não fornece uma URL
// de diretório confiável, então esse caso permanece explicitamente indisponível.
function resolveAsset(uri, artifactUrl) {
  if (!uri || !artifactUrl) return null;
  return new URL(uri, artifactUrl).href;
}

// Exibe a imagem que sustentou o claim e ancora visualmente o pixel e a box da
// região. O overlay continua sendo evidência 2D, não confirmação geométrica 3D.
function ObservationPreview({ observation, region, pixel, artifactUrl }) {
  const [showOverlay, setShowOverlay] = useState(true);
  if (!observation) return null;
  const uri = showOverlay ? observation.overlay_image_uri : observation.raw_image_uri;
  const source = resolveAsset(uri, artifactUrl);
  const box = region?.box;
  return (
    <div className="evidence">
      <div className="preview-controls">
        <strong>Frame de evidência</strong>
        <button type="button" onClick={() => setShowOverlay((value) => !value)}>
          {showOverlay ? "Ver imagem original" : "Ver regiões do VLM"}
        </button>
      </div>
      {source ? (
        <div className="preview-frame">
          <img src={source} alt="Observação RGB que sustenta o contexto selecionado" />
          {box && (
            <span
              className="region-box"
              style={{
                left: `${(100 * box.x_min) / observation.width}%`,
                top: `${(100 * box.y_min) / observation.height}%`,
                width: `${(100 * (box.x_max - box.x_min)) / observation.width}%`,
                height: `${(100 * (box.y_max - box.y_min)) / observation.height}%`,
              }}
            />
          )}
          {pixel && (
            <span
              className="pixel-marker"
              style={{
                left: `${(100 * pixel[0]) / observation.width}%`,
                top: `${(100 * pixel[1]) / observation.height}%`,
              }}
              title={`Pixel (${pixel[0]}, ${pixel[1]})`}
            />
          )}
        </div>
      ) : (
        <p>A preview requer que o artifact seja aberto pelo servidor.</p>
      )}
      <small>{observation.sensor_id} · {observation.frame_id} · {observation.timestamp_ns} ns</small>
    </div>
  );
}

// Resume os claims do frame para dar o contexto global daquele instante sem
// confundi-los com a classificação localizada da região selecionada.
function SceneClaims({ claims }) {
  if (!claims?.length) return null;
  return (
    <div>
      <h3>Contexto da cena</h3>
      <dl className="claims">
        {claims.map((claim) => (
          <React.Fragment key={`${claim.kind}:${claim.value}`}>
            <dt>{claim.kind}</dt><dd>{claim.value}</dd>
          </React.Fragment>
        ))}
      </dl>
    </div>
  );
}

// Apresenta a seleção por níveis de evidência: geometria, associação calibrada,
// claim visual e proveniência. Pontos sem contexto recebem diagnóstico direto.
function Inspector({ point, slice, artifactUrl }) {
  if (!point) return <p>Selecione um ponto.</p>;
  const association = point.association;
  if (!association) {
    return (
      <div>
        <p className="badge neutral">Sem observação visual</p>
        <p>Este ponto pertence ao mapa geométrico, mas nenhum frame processado foi associado a ele.</p>
        <h3>Geometria</h3>
        <code>{point.geometry_id}</code>
        <p>{point.coordinates_m.map((value) => value.toFixed(3)).join(", ")} m</p>
      </div>
    );
  }
  const observation = slice.observations?.find((item) => item.observation_id === association.rgb_observation_id);
  const region = slice.regions?.find((item) => item.region_id === association.region_id);
  const confidence = region?.confidence?.value;
  const supportState = region?.support?.state ?? "sem label de região";
  return (
    <div>
      <p className={`badge ${region ? "contextual" : "rgb"}`}>
        {region ? "Contexto VLM associado" : "RGB associado, sem região"}
      </p>
      <h3>{region?.label ?? "Pixel RGB observado"}</h3>
      {region && (
        <p>
          Confiança do VLM: <strong>{confidence == null ? "não informada" : `${(confidence * 100).toFixed(1)}%`}</strong>
          {" · "}suporte: <strong>{supportState}</strong>
          {region.geometric_confidence != null && ` · máscara: ${(region.geometric_confidence * 100).toFixed(1)}%`}
        </p>
      )}
      <ObservationPreview observation={observation} region={region} pixel={association.pixel} artifactUrl={artifactUrl} />
      {region?.claims?.length > 0 && (
        <div>
          <h3>Claims da região</h3>
          <ul className="claim-list">
            {region.claims.map((claim, index) => (
              <li key={`${claim.kind}:${claim.value}:${index}`}>
                <span>{claim.kind}</span> {claim.value}{claim.role && <small> · {claim.role}</small>}
              </li>
            ))}
          </ul>
        </div>
      )}
      <SceneClaims claims={observation?.scene_claims} />
      <details>
        <summary>Proveniência e dados técnicos</summary>
        <pre>{JSON.stringify(point, null, 2)}</pre>
        {region?.provenance && <pre>{JSON.stringify(region.provenance, null, 2)}</pre>}
      </details>
    </div>
  );
}

// Exibe o viewer persistido: carregamento do artifact, camadas independentes,
// órbita e seleção que recupera evidência e proveniência sem resetar a câmera.
function Explorer() {
  const [slice, setSlice] = useState(null);
  const [selected, setSelected] = useState(null);
  const [colorMode, setColorMode] = useState("geometry");
  const [artifactUrl, setArtifactUrl] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    const requestedUrl = new URLSearchParams(window.location.search).get("artifact");
    if (!requestedUrl) return undefined;
    const resolvedUrl = new URL(requestedUrl, window.location.href).href;
    const controller = new AbortController();
    fetch(resolvedUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`O servidor respondeu HTTP ${response.status}.`);
        return response.json();
      })
      .then((payload) => {
        const validated = validateSlice(payload);
        setSlice(validated);
        setSelected(null);
        setArtifactUrl(resolvedUrl);
        setColorMode(validated.schema_version === 2 ? "context" : "geometry");
        setError(null);
      })
      .catch((failure) => {
        if (failure.name !== "AbortError") {
          setError(failure instanceof Error ? failure.message : "Não foi possível abrir o artifact.");
        }
      });
    return () => controller.abort();
  }, []);
  const load = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      const parsed = validateSlice(JSON.parse(await file.text()));
      setSlice(parsed);
      setSelected(null);
      setArtifactUrl(null);
      setColorMode(parsed.schema_version === 2 ? "context" : "geometry");
      setError(null);
    } catch (failure) {
      setSlice(null);
      setSelected(null);
      setError(failure instanceof Error ? failure.message : "Não foi possível abrir o artifact.");
    }
  };
  return (
    <main>
      <header>
        <div><h1>Contextual 3D Map Explorer</h1><p>Geometria, RGB e contexto visual com evidência rastreável.</p></div>
        <input aria-label="Artifact do mapa" type="file" accept="application/json,.json" onChange={load} />
      </header>
      {error && <p role="alert" className="error">{error}</p>}
      {slice && (
        <>
          <div className="map-summary">
            <span>Mapa <code>{slice.map_id}</code></span>
            <span>{slice.points.length.toLocaleString("pt-BR")} pontos</span>
            <span>frame <code>{slice.map_frame}</code></span>
          </div>
          <Coverage slice={slice} />
          <nav className="layers" aria-label="Camada de coloração">
            {Object.entries(COLOR_MODES).map(([value, label]) => (
              <button type="button" key={value} className={colorMode === value ? "active" : ""}
                onClick={() => setColorMode(value)} disabled={value !== "geometry" && slice.schema_version < 2}>
                {label}
              </button>
            ))}
          </nav>
          <section>
            <div className="viewport">
              <Canvas camera={{ position: [0, -4, 2] }}>
                <color attach="background" args={["#0b0d12"]} />
                <Bounds fit clip margin={1.15}>
                  <Cloud points={slice.points} colorMode={colorMode} onSelect={setSelected} />
                  <SelectionMarker point={selected} />
                </Bounds>
                <OrbitControls makeDefault />
              </Canvas>
              <span className="viewport-hint">Arraste para orbitar · scroll para zoom · clique para inspecionar</span>
            </div>
            <aside><h2>Inspeção</h2><Inspector point={selected} slice={slice} artifactUrl={artifactUrl} /></aside>
          </section>
        </>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")).render(<Explorer />);
