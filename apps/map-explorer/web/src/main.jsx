import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Canvas } from "@react-three/fiber";
import { Bounds, OrbitControls, Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";
import "./styles.css";

// Converte o artifact público em buffers de renderização; o cliente nunca lê
// detalhes do backend de geometria ou associação.
function Cloud({ points, onSelect }) {
  const geometry = useMemo(() => {
    const positions = [];
    const colors = [];
    points.forEach((point) => {
      positions.push(...point.coordinates_m);
      const rgb = point.association?.color_rgb ?? point.display_color_rgb ?? [90, 90, 90];
      colors.push(...rgb.map((channel) => channel / 255));
    });
    const value = new THREE.BufferGeometry();
    value.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    value.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return value;
  }, [points]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <Points
      geometry={geometry}
      onClick={(event) => onSelect(points[event.index] ?? null)}
    >
      <PointMaterial vertexColors size={0.05} sizeAttenuation />
    </Points>
  );
}

// Valida somente a fronteira que o viewer consome, produzindo um diagnóstico
// legível quando o usuário seleciona outro JSON por engano.
function validateSlice(value) {
  if (value?.schema_version !== 1) {
    throw new Error("O artifact precisa usar schema_version 1.");
  }
  if (typeof value.map_id !== "string" || typeof value.map_frame !== "string") {
    throw new Error("O artifact precisa declarar map_id e map_frame.");
  }
  if (!Array.isArray(value.points)) {
    throw new Error("O artifact precisa conter uma lista points.");
  }
  value.points.forEach((point, index) => {
    if (!Array.isArray(point.coordinates_m) || point.coordinates_m.length !== 3) {
      throw new Error(`O ponto ${index} não possui coordinates_m tridimensional.`);
    }
  });
  return value;
}

// Exibe o primeiro viewer persistido: carregamento do artifact, órbita e
// seleção de ponto que revela a proveniência preservada na associação.
function Explorer() {
  const [slice, setSlice] = useState(null);
  const [selected, setSelected] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    const artifactUrl = new URLSearchParams(window.location.search).get("artifact");
    if (!artifactUrl) return undefined;
    const controller = new AbortController();
    fetch(artifactUrl, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`O servidor respondeu HTTP ${response.status}.`);
        return response.json();
      })
      .then((payload) => {
        setSlice(validateSlice(payload));
        setSelected(null);
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
      setError(null);
    } catch (failure) {
      setSlice(null);
      setSelected(null);
      setError(failure instanceof Error ? failure.message : "Não foi possível abrir o artifact.");
    }
  };
  return (
    <main>
      <h1>RGB–LiDAR Map Explorer</h1>
      <p>Abra um artifact JSON produzido pelo <code>mapping-runtime</code>.</p>
      <input
        aria-label="Artifact do mapa"
        type="file"
        accept="application/json,.json"
        onChange={load}
      />
      {error && <p role="alert" className="error">{error}</p>}
      {slice && (
        <>
          <p>
            Mapa <code>{slice.map_id}</code> no frame <code>{slice.map_frame}</code>:{" "}
            {slice.points.length} pontos.
          </p>
          <section>
            <Canvas camera={{ position: [0, -4, 2] }}>
              <color attach="background" args={["#101218"]} />
              <ambientLight intensity={1} />
              <Bounds fit clip observe margin={1.15}>
                <Cloud points={slice.points} onSelect={setSelected} />
              </Bounds>
              <OrbitControls makeDefault />
            </Canvas>
            <aside>
              <h2>Inspeção</h2>
              {selected ? <pre>{JSON.stringify(selected, null, 2)}</pre> : "Selecione um ponto."}
            </aside>
          </section>
        </>
      )}
    </main>
  );
}
createRoot(document.getElementById("root")).render(<Explorer />);
