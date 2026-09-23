# Documentation

One write-up per subsystem, plus the measurement ledger and the bug register.

| Subsystem | What it explains |
|---|---|
| [Data preparation](data-preparation.md) | SVG rasterisation, resize, normalization, and NPZ outputs. |
| [Model](model.md) | Generator/discriminator architecture and actual input signatures. |
| [Training](training.md) | Batch selection, discriminator/generator updates, checkpoints, and wiring measurements. |
| [Inference](inference.md) | Generation-data preparation, model loading, and the currently failing prediction path. |
| [Measurement](measurement.md) | Commands and provenance for every number and image in this pass. |
| [Bugs found](BUGS-FOUND.md) | Reproductions and proposed fixes without changing the implementation. |

[← back to the overview](../README.md)
