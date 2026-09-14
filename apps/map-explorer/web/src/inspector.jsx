import React, { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { observationIdOf, visualEvidence } from "./map-data.js";

// Traduz o motivo pelo qual um ponto do mapa ficou sem evidência contextual.
// Sem isso o viewer só sabe dizer que nada foi publicado, e a pergunta que o
// usuário faz diante de um ponto cinza ao lado de um colorido é "por quê".
// Explica o estado de corroboração em vez de mostrar só um rótulo técnico. O
// usuário precisa saber por que um ponto está apagado no mapa.
const SUPPORT_STATE_COPY = Object.freeze({
  weak: "A vizinhança geométrica e os outros keyframes discordam deste label. O ponto aparece apagado no mapa.",
  uncorroborated: "Um único keyframe observou este ponto, então não há verificação cruzada possível.",
  corroborated: "A vizinhança geométrica ou os demais keyframes sustentam este label.",
});

const MISSING_CONTEXT_REASON = Object.freeze({
  behind_camera: "Ficou fora do campo de visão de todos os keyframes.",
  outside_image: "Projetou-se fora dos limites da imagem.",
  outside_valid_support: "Caiu fora do suporte óptico válido da lente.",
  occluded: "Ficou atrás de uma superfície mais próxima da câmera.",
  visibility_unconfirmed: "A geometria medida não sustenta uma primeira superfície compatível com este ponto.",
});

const SURFACE_REASON_COPY = Object.freeze({
  first_surface_match: "O ponto coincide com a primeira superfície medida.",
  behind_first_surface: "Outra superfície medida intercepta o raio antes deste ponto.",
  point_before_measured_surface: "O ponto está antes do suporte que foi possível reconstruir.",
  no_measured_surface: "Não há suporte medido suficiente neste raio.",
  contour_anchored_surface: "O contorno da região sustenta este componente 3D.",
  background_behind_contour: "A superfície está atrás do contorno da região.",
  insufficient_surface_anchor: "O componente não tem âncora geométrica suficiente no contorno.",
  no_matching_raster_surface: "O pixel da máscara não confirma a mesma superfície do ponto.",
});

// Explica separadamente o alcance do raio e o vínculo semântico. Uma paisagem
// através da janela pode ter RGB válido e continuar sem o label da janela.
function SurfaceEvidence({ evidence }) {
  const surface = evidence?.surface_evidence;
  if (!surface) return null;
  const meters = (value) => value == null ? "Sem suporte medido" : `${value.toFixed(3)} m`;
  return (
    <div className="inspector-block">
      <h3>Visibilidade e superfície</h3>
      <div className="classification-grid">
        <span>Distância do ponto</span><strong>{meters(surface.point_depth_m)}</strong>
        <span>Primeira superfície</span><strong>{meters(surface.first_surface_depth_m)}</strong>
        <span>Tolerância geométrica</span><strong>{meters(surface.tolerance_m)}</strong>
        <span>Vínculo do rótulo</span><strong>{surface.surface_support === "supported" ? "Sustentado em 3D" : surface.surface_support === "uncertain" ? "Incerto · hipótese 2D" : "Não avaliado"}</strong>
      </div>
      <p className="empty-copy">{SURFACE_REASON_COPY[surface.reason] ?? surface.reason}</p>
      {surface.support_reason && <p className="empty-copy">{SURFACE_REASON_COPY[surface.support_reason] ?? surface.support_reason}</p>}
    </div>
  );
}

// Resolve um asset relativo ao JSON servido. Upload local não fornece uma URL
// de diretório confiável, então esse caso permanece explicitamente indisponível.
function resolveAsset(uri, artifactUrl) {
  if (!uri || !artifactUrl) return null;
  return new URL(uri, artifactUrl).href;
}

// Mantém imagem, box e pixel no mesmo sistema de coordenadas em qualquer
// tamanho. Existe porque ``object-fit`` deslocaria as marcações nas faixas
// vazias quando o aspect ratio do container divergisse do frame.
function ObservationImage({ source, observation, region, pixel, marked }) {
  const box = region?.box;
  const markerRadius = Math.max(observation.width, observation.height) * 0.006;
  return (
    <svg
      className="observation-image"
      viewBox={`0 0 ${observation.width} ${observation.height}`}
      role="img"
      aria-label={marked
        ? "Observação RGB com as marcações que sustentam o contexto selecionado"
        : "Observação RGB original, sem marcações"}
    >
      <image href={source} width={observation.width} height={observation.height} />
      {marked && box && (
        <rect
          className="region-box"
          x={box.x_min}
          y={box.y_min}
          width={box.x_max - box.x_min}
          height={box.y_max - box.y_min}
          vectorEffect="non-scaling-stroke"
        />
      )}
      {marked && pixel && (
        <circle
          className="pixel-marker"
          cx={pixel[0]}
          cy={pixel[1]}
          r={markerRadius}
          vectorEffect="non-scaling-stroke"
        >
          <title>{`Pixel (${pixel[0]}, ${pixel[1]})`}</title>
        </circle>
      )}
    </svg>
  );
}

// Mantém a navegação por Tab dentro do modal enquanto ele estiver aberto.
// Existe para que o diálogo modal não entregue foco aos controles encobertos
// do mapa ou do inspector.
function keepDialogFocus(event) {
  if (event.key !== "Tab") return;
  const controls = [...event.currentTarget.querySelectorAll("button:not([disabled])")];
  if (!controls.length) return;
  const first = controls[0];
  const last = controls.at(-1);
  if (event.shiftKey && (document.activeElement === first || document.activeElement === event.currentTarget)) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

// Exibe a imagem que sustentou o claim e ancora visualmente o pixel e a box da
// região. O overlay continua sendo evidência 2D, não confirmação geométrica 3D.
function ObservationPreview({ observation, region, pixel, artifactUrl }) {
  const [showOverlay, setShowOverlay] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const expandButton = useRef(null);
  const dialog = useRef(null);
  useEffect(() => {
    setShowOverlay(true);
    setExpanded(false);
  }, [observation?.observation_id, region?.region_id]);
  useEffect(() => {
    if (!expanded) return undefined;
    dialog.current?.focus();
    return () => expandButton.current?.focus();
  }, [expanded]);
  if (!observation) return null;
  const uri = showOverlay ? observation.overlay_image_uri : observation.raw_image_uri;
  const source = resolveAsset(uri, artifactUrl);
  const toggleOverlay = () => setShowOverlay((value) => !value);
  const closeExpanded = () => setExpanded(false);
  return (
    <div className="evidence-block">
      <div className="block-heading">
        <strong>Frame · {showOverlay ? "Regiões VLM" : "Original"}</strong>
        <div className="block-actions">
          <button className="text-action" type="button" onClick={toggleOverlay}>
            {showOverlay ? "Ver original" : "Ver regiões"}
          </button>
          {source && (
            <button
              className="text-action"
              type="button"
              ref={expandButton}
              onClick={() => setExpanded(true)}
              aria-haspopup="dialog"
            >
              Ampliar
            </button>
          )}
        </div>
      </div>
      {source ? (
        <div className="preview-frame" style={{ aspectRatio: `${observation.width} / ${observation.height}` }}>
          <ObservationImage source={source} observation={observation} region={region} pixel={pixel} marked={showOverlay} />
        </div>
      ) : (
        <p className="empty-copy">A preview requer que o artifact seja aberto pelo servidor.</p>
      )}
      {expanded && createPortal(
        <div
          className="image-dialog-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) closeExpanded();
          }}
        >
          <section
            className="image-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="expanded-frame-title"
            tabIndex={-1}
            ref={dialog}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                closeExpanded();
              } else keepDialogFocus(event);
            }}
          >
            <header className="image-dialog-heading">
              <strong id="expanded-frame-title">Frame · {showOverlay ? "Regiões VLM" : "Original"}</strong>
              <div className="block-actions">
                <button className="text-action" type="button" onClick={toggleOverlay}>
                  {showOverlay ? "Ver original" : "Ver regiões"}
                </button>
                <button className="dialog-close" type="button" onClick={closeExpanded} aria-label="Fechar imagem ampliada">×</button>
              </div>
            </header>
            <div className="image-dialog-stage">
              <ObservationImage source={source} observation={observation} region={region} pixel={pixel} marked={showOverlay} />
            </div>
          </section>
        </div>,
        document.body,
      )}
      <div className="technical-line">
        <span>{observation.sensor_id}</span><span>{observation.frame_id}</span>
      </div>
    </div>
  );
}

// Resume os claims do frame para dar contexto global sem confundi-los com a
// classificação localizada da região selecionada.
function SceneClaims({ claims }) {
  if (!claims?.length) return null;
  return (
    <div className="inspector-block">
      <h3>Contexto da cena</h3>
      <dl className="claims-grid">
        {claims.map((claim) => (
          <React.Fragment key={`${claim.kind}:${claim.value}`}>
            <dt>{claim.kind}</dt><dd>{claim.value}</dd>
          </React.Fragment>
        ))}
      </dl>
    </div>
  );
}

// Mostra quantas observações classificaram o mesmo ponto e o quanto elas
// concordam. Com vários keyframes, a concordância é evidência: um label
// sustentado por cinco frames não tem o mesmo peso de um sustentado por um.
function FusionSummary({ evidence, activeContribution, onSelectContribution }) {
  const contributions = evidence?.contributions ?? [];
  const state = evidence?.support_state;
  if (contributions.length < 2 && !state && !evidence?.consolidation) return null;
  return (
    <div className="inspector-block">
      <h3>Sustentação do label</h3>
      <div className="classification-grid">
        {evidence?.consolidation?.resolution === "quality_tiebreak" && <><span>Decisão</span><strong>Desempate por qualidade</strong></>}
        {evidence?.consolidation?.resolution === "majority" && <><span>Decisão</span><strong>Maioria entre runs</strong></>}
        {state && <><span>Estado</span><strong>{state}</strong></>}
        <span>{evidence?.consolidation ? "Runs com evidência" : "Observações"}</span><strong>{Math.max(contributions.length, 1)}</strong>
        <span>{evidence?.consolidation ? "Concordância entre runs" : "Concordância entre frames"}</span>
        <strong>{evidence.agreement == null ? "Sem corroboração" : `${(evidence.agreement * 100).toFixed(0)}%`}</strong>
        <span>Suporte da vizinhança 3D</span>
        <strong>{evidence.spatial_support == null ? "Indefinido" : `${(evidence.spatial_support * 100).toFixed(0)}%`}</strong>
      </div>
      {state && <p className="empty-copy">{SUPPORT_STATE_COPY[state]}</p>}
      <ul className="claim-list">
        {contributions.length > 1 && contributions.map((item) => (
          <li key={`${item.observation_id}:${item.region_id}`}>
            <button className="text-action" type="button" onClick={() => onSelectContribution(item)}
              aria-pressed={activeContribution?.observation_id === item.observation_id && activeContribution?.region_id === item.region_id}>
              {item.source_run_id ?? item.observation_id}
            </button>
            <strong>{item.raw_label ?? item.label}</strong>
            {item.confidence != null && <small>{(item.confidence * 100).toFixed(0)}%</small>}
          </li>
        ))}
      </ul>
    </div>
  );
}

// Apresenta a seleção em camadas de evidência e mantém metadados extensos num
// disclosure técnico, reduzindo a densidade inicial do dock.
export function Inspector({
  point,
  slice,
  artifactUrl,
  onClose,
  onFocus,
  outOfFocus = false,
}) {
  const evidence = point ? visualEvidence(point) : null;
  const [activeContribution, setActiveContribution] = useState(null);
  useEffect(() => setActiveContribution(null), [point?.geometry_id]);
  const displayedEvidence = activeContribution ?? evidence;
  const observation = slice.observations?.find(
    (item) => item.observation_id === observationIdOf(displayedEvidence),
  );
  const region = slice.regions?.find((item) => item.region_id === displayedEvidence?.region_id);
  const confidence = region?.confidence?.value;
  const supportState = region?.support?.state;
  return (
    <aside className="inspector-dock" aria-label="Detalhes do mapa">
      <div className="dock-heading">
        <span>Detalhes</span>
        <div>{point && <button type="button" title="Focar seleção" onClick={onFocus}>Focar</button>}<button type="button" aria-label="Recolher detalhes" onClick={onClose}>×</button></div>
      </div>
      <div className="inspector-scroll">
        {!point ? (
          <div className="empty-inspector">
            <strong>Nenhum ponto selecionado</strong>
            <p>Clique na nuvem para consultar contexto e evidência.</p>
          </div>
        ) : (
          <>
            {outOfFocus && (
              <div className="filter-warning">O ponto selecionado está fora do foco atual da legenda.</div>
            )}
            <div className="selection-title">
              <span className={`status-tag ${evidence?.label ? "contextual" : "neutral"}`}>
                {region
                  ? evidence?.label ? "Evidência VLM" : "Hipótese 2D · sem rótulo 3D"
                  : evidence?.status === "associated"
                    ? "Sem evidência publicada"
                    : "Não observado"}
              </span>
              <h2>{evidence?.label ?? (evidence?.tentative_label ? `Hipótese: ${evidence.tentative_label}` : "Ponto sem evidência contextual")}</h2>
              <code>{point.geometry_id}</code>
            </div>

            <div className="metric-strip">
              <div><small>X</small><strong>{point.coordinates_m[0].toFixed(2)} m</strong></div>
              <div><small>Y</small><strong>{point.coordinates_m[1].toFixed(2)} m</strong></div>
              <div><small>Z</small><strong>{point.coordinates_m[2].toFixed(2)} m</strong></div>
            </div>

            {!region && (
              <div className="inspector-block warning-block">
                <strong>Sem evidência contextual publicada</strong>
                <p>
                  {MISSING_CONTEXT_REASON[evidence?.status]
                    ?? (evidence
                      ? "A câmera observou este ponto, mas nenhuma evidência contextual publicada o cobriu."
                      : "Nenhum keyframe deste trecho considerou este ponto.")}
                </p>
              </div>
            )}

            {region && (
              <div className="inspector-block">
                <h3>Classificação</h3>
                <div className="classification-grid">
                  <span>Confiança VLM</span><strong>{confidence == null ? "Não informada" : `${(confidence * 100).toFixed(1)}%`}</strong>
                  <span>Suporte</span><strong>{supportState ?? "Não informado"}</strong>
                  <span>Qualidade da máscara</span><strong>{region.geometric_confidence == null ? "Não informada" : `${(region.geometric_confidence * 100).toFixed(1)}%`}</strong>
                </div>
              </div>
            )}

            <SurfaceEvidence evidence={displayedEvidence} />

            {evidence && (
              <ObservationPreview observation={observation} region={region} pixel={displayedEvidence.pixel} artifactUrl={artifactUrl} />
            )}

            <FusionSummary evidence={evidence} activeContribution={activeContribution} onSelectContribution={setActiveContribution} />

            {region?.claims?.length > 0 && (
              <div className="inspector-block">
                <h3>Claims da região</h3>
                <ul className="claim-list">
                  {region.claims.map((claim, index) => (
                    <li key={`${claim.kind}:${claim.value}:${index}`}>
                      <span>{claim.kind}</span><strong>{claim.value}</strong>
                      {claim.role && <small>{claim.role}</small>}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <SceneClaims claims={observation?.scene_claims} />
            <details className="technical-details">
              <summary>Proveniência e dados técnicos</summary>
              <pre>{JSON.stringify(point, null, 2)}</pre>
              {region?.provenance && <pre>{JSON.stringify(region.provenance, null, 2)}</pre>}
            </details>
          </>
        )}
      </div>
    </aside>
  );
}
