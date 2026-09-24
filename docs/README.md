# Documentation

One write-up per subsystem, plus the measurement ledger.

| Subsystem | What it explains |
|---|---|
| [Data preparation](data-preparation.md) | SVG rasterisation, resize, normalization, and NPZ outputs. |
| [Model](model.md) | Generator/discriminator architecture and actual input signatures. |
| [Training](training.md) | Batch and metadata selection, discriminator/generator updates, checkpoints, wiring, and one-epoch timing. |
| [Inference](inference.md) | Generation-data preparation, model loading, and conditioned prediction. |
| [Measurement](measurement.md) | The Linux container run behind every number and image. |

[← back to the overview](../README.md)
