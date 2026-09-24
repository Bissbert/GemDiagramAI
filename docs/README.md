# Documentation

One write-up per subsystem, plus the measurement ledger and the bug register.

| Subsystem | What it explains |
|---|---|
| [Data preparation](data-preparation.md) | SVG rasterisation, resize, normalization, and NPZ outputs. |
| [Model](model.md) | Generator/discriminator architecture and actual input signatures. |
| [Training](training.md) | Batch selection, discriminator/generator updates, checkpoints, wiring, and the first-epoch failure. |
| [Inference](inference.md) | Generation-data preparation, model loading, and the currently failing prediction path. |
| [Measurement](measurement.md) | The Linux container run behind every number and image. |
| [Bugs found](BUGS-FOUND.md) | Each bug found, with its reproduction and whether it is fixed, open, or not a bug. |

[← back to the overview](../README.md)
