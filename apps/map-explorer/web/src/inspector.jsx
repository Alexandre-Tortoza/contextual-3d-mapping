import React, { useEffect, useState } from "react";

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
        <strong>Frame de evidência</strong>
        <button className="text-action" type="button" onClick={() => setShowOverlay((value) => !value)}>
          {showOverlay ? "Imagem original" : "Regiões do VLM"}
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

// Apresenta a seleção em camadas de evidência e mantém metadados extensos num
// disclosure técnico, reduzindo a densidade inicial do dock.
export function Inspector({
  point,
  slice,
  artifactUrl,
  onClose,
  onFocus,
  hiddenByFilter = false,
}) {
  const association = point?.association;
  const observation = slice.observations?.find(
    (item) => item.observation_id === association?.rgb_observation_id,
  );
  const region = slice.regions?.find((item) => item.region_id === association?.region_id);
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
            {hiddenByFilter && (
              <div className="filter-warning">O ponto selecionado está oculto pelo filtro da legenda.</div>
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
                <p>Este ponto ainda não possui um label produzido pelo pipeline visual.</p>
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

            {association && (
              <ObservationPreview observation={observation} region={region} pixel={association.pixel} artifactUrl={artifactUrl} />
            )}

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
