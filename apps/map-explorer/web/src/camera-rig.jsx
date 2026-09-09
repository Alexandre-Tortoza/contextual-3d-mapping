import React, { forwardRef, useEffect, useImperativeHandle, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { CameraControls, CameraControlsImpl } from "@react-three/drei";
import * as THREE from "three";

import { cameraPreset, isEditableTarget, movementFromKeys, navigationScale } from "./navigation.js";

// Controla a câmera como estado independente da nuvem, da seleção e dos
// filtros. Assim apenas comandos explícitos alteram o centro de interesse.
export const CameraRig = forwardRef(function CameraRig(
  { metrics, flyMode, mapKey, reducedMotion = false },
  forwardedRef,
) {
  const controlsRef = useRef(null);
  const pressedKeys = useRef(new Set());
  const scale = useMemo(() => navigationScale(metrics.diagonal), [metrics.diagonal]);

  // Expõe somente os comandos necessários à toolbar e aos atalhos, mantendo a
  // implementação concreta de câmera dentro do viewport.
  useImperativeHandle(forwardedRef, () => ({
    reset() {
      if (!controlsRef.current) return undefined;
      const pose = cameraPreset(metrics, "isometric");
      return controlsRef.current.setLookAt(...pose.position, ...pose.target, !reducedMotion);
    },
    focus(point) {
      if (!controlsRef.current || !point) return undefined;
      const position = controlsRef.current.getPosition(new THREE.Vector3());
      const target = controlsRef.current.getTarget(new THREE.Vector3());
      const direction = position.sub(target);
      if (direction.lengthSq() === 0) direction.set(1, -1, 0.65);
      direction.normalize().multiplyScalar(scale.focusDistance);
      const [x, y, z] = point.coordinates_m;
      return controlsRef.current.setLookAt(
        x + direction.x,
        y + direction.y,
        z + direction.z,
        x,
        y,
        z,
        !reducedMotion,
      );
    },
    preset(name) {
      if (!controlsRef.current) return undefined;
      const pose = cameraPreset(metrics, name);
      return controlsRef.current.setLookAt(
        ...pose.position,
        ...pose.target,
        !reducedMotion,
      );
    },
  }), [metrics, reducedMotion, scale.focusDistance]);

  // Faz o enquadramento uma única vez por mapa aberto. Alterações posteriores
  // de seleção, camada ou filtro não reexecutam este efeito.
  useEffect(() => {
    const pose = cameraPreset(metrics, "isometric");
    controlsRef.current?.setLookAt(...pose.position, ...pose.target, false);
  }, [mapKey, metrics]);

  // Mantém o conjunto de teclas pressionadas somente durante modo voo e limpa
  // o estado ao perder foco para impedir movimento preso.
  useEffect(() => {
    if (!flyMode) {
      pressedKeys.current.clear();
      return undefined;
    }
    const press = (event) => {
      if (isEditableTarget(event.target)) return;
      if (!event.repeat) pressedKeys.current.add(event.code);
    };
    const release = (event) => pressedKeys.current.delete(event.code);
    const clear = () => pressedKeys.current.clear();
    window.addEventListener("keydown", press);
    window.addEventListener("keyup", release);
    window.addEventListener("blur", clear);
    return () => {
      window.removeEventListener("keydown", press);
      window.removeEventListener("keyup", release);
      window.removeEventListener("blur", clear);
      clear();
    };
  }, [flyMode]);

  // Move câmera e alvo juntos a cada frame no modo voo; isso elimina o alvo
  // fixo sem abandonar a orientação controlada pelo mouse.
  useFrame((_, delta) => {
    if (!flyMode || !controlsRef.current || pressedKeys.current.size === 0) return;
    const accelerated = pressedKeys.current.has("ShiftLeft") || pressedKeys.current.has("ShiftRight");
    const movement = movementFromKeys(pressedKeys.current, Math.min(delta, 0.05), scale.flySpeed, accelerated);
    if (movement.horizontal) controlsRef.current.truck(movement.horizontal, 0, false);
    if (movement.vertical) controlsRef.current.elevate(movement.vertical, false);
    if (movement.forward) controlsRef.current.forward(movement.forward, false);
  });

  return (
    <CameraControls
      ref={controlsRef}
      makeDefault
      dollyToCursor
      infinityDolly
      smoothTime={reducedMotion ? 0 : 0.18}
      draggingSmoothTime={reducedMotion ? 0 : 0.08}
      truckSpeed={2}
      mouseButtons={{
        left: CameraControlsImpl.ACTION.ROTATE,
        middle: CameraControlsImpl.ACTION.TRUCK,
        right: CameraControlsImpl.ACTION.TRUCK,
        wheel: CameraControlsImpl.ACTION.DOLLY,
      }}
      touches={{
        one: CameraControlsImpl.ACTION.TOUCH_ROTATE,
        two: CameraControlsImpl.ACTION.TOUCH_DOLLY_TRUCK,
        three: CameraControlsImpl.ACTION.TOUCH_TRUCK,
      }}
    />
  );
});
