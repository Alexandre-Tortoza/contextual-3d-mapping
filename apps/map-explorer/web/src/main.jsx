import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Points, PointMaterial } from "@react-three/drei";
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
      const rgb = point.association?.color_rgb ?? [90, 90, 90];
      colors.push(...rgb.map((channel) => channel / 255));
    });
    const value = new THREE.BufferGeometry();
    value.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    value.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    return value;
  }, [points]);
  return <Points geometry={geometry} onClick={(event) => onSelect(points[event.index] ?? null)}><PointMaterial vertexColors size={0.05} sizeAttenuation /></Points>;
}

// Exibe o primeiro viewer persistido: carregamento do artifact, órbita e
// seleção de ponto que revela a proveniência preservada na associação.
function Explorer() {
  const [slice, setSlice] = useState(null);
  const [selected, setSelected] = useState(null);
  const load = async (event) => {
    const text = await event.target.files[0]?.text();
    if (text) { setSlice(JSON.parse(text)); setSelected(null); }
  };
  return <main><h1>RGB–LiDAR Map Explorer</h1><input aria-label="Artifact do mapa" type="file" accept="application/json" onChange={load} />
    {slice && <><p>Mapa <code>{slice.map_id}</code> no frame <code>{slice.map_frame}</code>: {slice.points.length} pontos.</p>
      <section><Canvas camera={{ position: [0, -4, 2] }}><color attach="background" args={["#101218"]} /><ambientLight intensity={1} /><Cloud points={slice.points} onSelect={setSelected} /><OrbitControls makeDefault /></Canvas>
      <aside><h2>Inspeção</h2>{selected ? <pre>{JSON.stringify(selected, null, 2)}</pre> : "Selecione um ponto."}</aside></section></>}
  </main>;
}
createRoot(document.getElementById("root")).render(<Explorer />);
