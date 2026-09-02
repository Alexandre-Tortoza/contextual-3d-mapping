# Observation Contracts

Shared observation identity, source references, provenance, evidence references, and cross-sensor metadata belong here when they are used across multiple modules or applications.

Sensor- or capability-specific payloads should remain owned by their defining module or adapter.

`contextual_mapping_contracts` defines stable observation, source artifact and provenance
references. Provenance retains every contributing observation, supporting one-to-many and
many-to-one evidence without overwriting contributors.
