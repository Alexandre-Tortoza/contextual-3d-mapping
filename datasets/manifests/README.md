# Dataset Manifests

Manifests de dataset e sequência descrevendo identidade de origem, disponibilidade de sensor, localizações de artifact, convenções de coordenadas, referências de calibração e metadados de reprodutibilidade.

Cada arquivo `datasets/manifests/<dataset-id>.json` descreve o dataset local em
`datasets/raw/<dataset-id>/`. O arquivo é rastreado pelo Git; seus artifacts são
caminhos relativos àquela raiz e os dados brutos continuam ignorados.

O loader público `contextual_mapping_datasets.load_dataset_manifest` valida o schema
`1.0`, ids únicos, clocks, frames, calibrações e referências de artifacts antes que
um adapter de formato tente ler qualquer payload.

Estrutura mínima:

```json
{
  "dataset_id": "example-dataset",
  "schema_version": "1.0",
  "source_uri": "https://example.org/dataset",
  "sequences": [
    {
      "sequence_id": "sequence-01",
      "split": "evaluation",
      "calibrations": [
        {
          "calibration_id": "camera-to-body",
          "artifact_uri": "calibration/camera.json",
          "source_frame": "camera",
          "target_frame": "body"
        }
      ],
      "sensors": [
        {
          "sensor_id": "camera",
          "kind": "rgb",
          "artifact_uri": "sequence-01/images",
          "media_type": "image/png",
          "frame_id": "camera",
          "clock_id": "dataset-clock",
          "calibration_id": "camera-to-body",
          "required": true
        }
      ]
    }
  ]
}
```

O `artifact_uri` pode apontar para arquivo ou diretório; o parser concreto do
dataset define como enumerá-lo. Um `artifact_uri` absoluto, uma URL ou qualquer
segmento `.`/`..` é inválido.
