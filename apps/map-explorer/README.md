# Map Explorer

`map-explorer` is the primary human-facing application for opening, querying, and visually inspecting persistent maps.

## Responsibilities

- open a map through stable application contracts;
- render 3D geometry and semantic overlays;
- submit semantic, spatial, relational, and contextual queries through `query-engine`;
- focus the viewer on returned entities or regions;
- expose related observations, evidence, confidence, and provenance;
- visualize scene-graph relations without owning graph construction.

## Initial structure

```text
map-explorer/
├── README.md
├── api/
├── web/
└── configs/
```

The API and frontend are delivery layers. They must not directly depend on private module implementations or storage-specific schemas.
