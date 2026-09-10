import React, { useEffect, useState } from "react";

import { observationIdOf, visualEvidence } from "./map-data.js";

// Traduz o motivo pelo qual um ponto do mapa ficou sem contexto. Sem isso o
// viewer só sabe dizer que falta label, e a pergunta que o usuário faz diante
// de um ponto cinza ao lado de um colorido é justamente "por quê".
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
});

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
  useEffect(() => setShowOverlay(true), [observation?.observation_id, region?.region_id]);
  if (!observation) return null;
  const uri = showOverlay ? observation.overlay_image_uri : observation.raw_image_uri;
  const source = resolveAsset(uri, artifactUrl);
  const box = region?.box;
  return (
    <div className="evidence-block">
      <div className="block-heading">
        <strong>Frame · {showOverlay ? "Regiões VLM" : "Original"}</strong>
        <button className="text-action" type="button" onClick={() => setShowOverlay((value) => !value)}>
          {showOverlay ? "Ver original" : "Ver regiões"}
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
        <p className="empty-copy">A preview requer que o artifact seja aberto pelo servidor.</p>
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
function FusionSummary({ evidence }) {
  const contributions = evidence?.contributions ?? [];
  const state = evidence?.support_state;
  if (contributions.length < 2 && !state) return null;
  return (
    <div className="inspector-block">
      <h3>Sustentação do label</h3>
      <div className="classification-grid">
        {state && <><span>Estado</span><strong>{state}</strong></>}
        <span>Observações</span><strong>{Math.max(contributions.length, 1)}</strong>
        <span>Concordância entre frames</span>
        <strong>{evidence.agreement == null ? "Sem corroboração" : `${(evidence.agreement * 100).toFixed(0)}%`}</strong>
        <span>Suporte da vizinhança 3D</span>
        <strong>{evidence.spatial_support == null ? "Indefinido" : `${(evidence.spatial_support * 100).toFixed(0)}%`}</strong>
      </div>
      {state && <p className="empty-copy">{SUPPORT_STATE_COPY[state]}</p>}
      <ul className="claim-list">
        {contributions.length > 1 && contributions.map((item) => (
          <li key={`${item.observation_id}:${item.region_id}`}>
            <span>{item.observation_id}</span><strong>{item.label}</strong>
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
  const observation = slice.observations?.find(
    (item) => item.observation_id === observationIdOf(evidence),
  );
  const region = slice.regions?.find((item) => item.region_id === evidence?.region_id);
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
              <span className={`status-tag ${region ? "contextual" : "neutral"}`}>
                {region ? "Contexto VLM" : "Sem contexto"}
              </span>
              <h2>{region?.label ?? "Ponto sem contexto"}</h2>
              <code>{point.geometry_id}</code>
            </div>

            <div className="metric-strip">
              <div><small>X</small><strong>{point.coordinates_m[0].toFixed(2)} m</strong></div>
              <div><small>Y</small><strong>{point.coordinates_m[1].toFixed(2)} m</strong></div>
              <div><small>Z</small><strong>{point.coordinates_m[2].toFixed(2)} m</strong></div>
            </div>

            {!region && (
              <div className="inspector-block warning-block">
                <strong>Sem classificação contextual</strong>
                <p>
                  {MISSING_CONTEXT_REASON[evidence?.status]
                    ?? (evidence
                      ? "A câmera observou este ponto, mas nenhuma região do frame o cobriu."
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

            {evidence && (
              <ObservationPreview observation={observation} region={region} pixel={evidence.pixel} artifactUrl={artifactUrl} />
            )}

            <FusionSummary evidence={evidence} />

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
