# Artifacts

## Canonical serialization (issue #172)

`infrastructure/serialization.py` serializes a `VisualObservation` to a plain JSON-able
dict and back, round-tripping without information loss:

- masks are embedded as compact run-length-encoded booleans — small, exact, and
  self-contained, since a region's own geometry is required to interpret the observation
  at all;
- visual/language embeddings are referenced by id only
  (`visual_embedding_ref`/`language_embedding_ref`); the vectors themselves are not part
  of the canonical observation (see [persistence integration](#persistence));
  `Evidence.artifact` references raw model evidence the same way, through the shared
  `SourceArtifactReference`;
- `schema_version` and `coordinate_convention` travel with every payload;
  `UnsupportedSchemaVersionError` fails predictably on a version this module cannot read.

## Persistence

`infrastructure/integration/persistence_integration.py` (#180) defines the
`EvidencePersistencePort` this module needs from repository persistence (#112) — an
opaque `put(id, payload) -> reference` / `get(reference) -> payload` pair — and:

- `persist_observation` / `reload_observation` round-trip a full `VisualObservation`;
- `persist_visual_embeddings` / `persist_language_embeddings` persist the pooled
  embeddings a pipeline run produced (these live outside the canonical observation
  itself, referenced only by id).

No storage-specific client or path appears in any public module contract; tests exercise
this boundary with `infrastructure/fakes/fake_evidence_store.InMemoryEvidenceStore`.

## Stage cache artifacts

See [execution.md](execution.md#stage-cache-issue-170): `application/cache.StageCache`
writes one JSON record per `(stage_name, fingerprint)` under a caller-chosen cache
directory, independent of the canonical serialization format above.
