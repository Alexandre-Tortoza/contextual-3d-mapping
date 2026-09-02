# Artifacts

Os artefatos específicos das pipelines `baseline` (`vlm_*.json`, `detections.json`,
masks, overlays, `result.json`) e `region-first` (`semantic_regions.json`,
`discovered_regions.json`, embeddings, `region_overlay.png`, `comparison.json`) foram
removidos junto com o código que os produzia (issue
[#4](https://github.com/Alexandre-Tortoza/image-context/issues/4)).

## O que `sample` persiste hoje

```text
runs/<run-id>/
├── manifest.json
└── selected_frames.json
└── frames/
    └── frame-XXXXXX/
        └── image.png
```

- `manifest.json`: fingerprint da configuração de amostragem (`dataset.*`); usado para
  detectar reuso incompatível de um `run-id`.
- `selected_frames.json`: metadados de cada imagem extraída (`frame_id`,
  `source_index`, `timestamp_ns`, dimensões, caminho).
- `frames/<frame-id>/image.png`: imagem decodificada do ROS bag.

O contrato de artefatos do módulo canônico (`ImageContext` com regiões, embeddings,
relações, incerteza e proveniência) está sendo desenhado na issue
[#5](https://github.com/Alexandre-Tortoza/image-context/issues/5); esta página será
reescrita quando esse contrato for implementado.
