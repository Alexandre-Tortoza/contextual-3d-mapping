# Dataset Adapters

Shared adapters that translate supported dataset layouts into repository input contracts.

Dataset-specific parsing, timestamp normalization, calibration loading, frame naming, and artifact resolution belong here when reused across applications or modules.

The public `contextual_mapping_adapters` package exposes a narrow
`MultimodalDatasetAdapter` protocol and payload-independent `CanonicalObservation` values.
Every observation preserves source identity, integer nanosecond timestamp, clock, frame,
sensor, calibration and external artifact reference. Invalid manifest metadata fails before
iteration.

`synchronize` creates one group per configured anchor observation. Input order is ignored; the
closest unused observation of each expected kind is selected within an inclusive tolerance.
Ties resolve by timestamp, sequence index and observation id. Original timestamps are preserved
and missing kinds are explicit. A call accepts exactly one sequence and clock; interpretation,
interpolation and fusion remain downstream responsibilities.
