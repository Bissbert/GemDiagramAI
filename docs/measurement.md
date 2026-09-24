# How this was measured

[← back to the overview](../README.md)

Every number and raster image in this documentation comes from one script run
in a Linux container:

```sh
sh tools/linux-run.sh training-data > media/captures/linux-run.txt
```

[`tools/linux-run.sh`](../tools/linux-run.sh) starts `python:3.11-slim-bookworm`,
mounts the repository and the prepared dataset read-only, copies the repository,
installs `requirements.txt` unmodified, runs the test suite, and runs each
script in [`tools/`](../tools). The
four PNGs in `media/` are written back through a second mount. The full output
is [`media/captures/linux-run.txt`](../media/captures/linux-run.txt); every
block below is taken from it.

`training-data/` is git-ignored. It holds the 4,992 diagrams the author
prepared earlier. Without it the script skips the full-dataset measurement and
the two contact sheets.

## Environment

| | |
|---|---|
| Kernel | Linux 6.5.11-linuxkit, aarch64 (Docker Desktop VM, 10 CPUs, 33 GB memory) |
| Image | `python:3.11-slim-bookworm` (`sha256:a36c24f9…4fec56b`), Python 3.11.16 |
| Packages | TensorFlow 2.14.0 (CPU), numpy 1.26.4, Pillow 12.3.0, CairoSVG 2.9.1, matplotlib 3.11.2 |
| System packages | `libcairo2`, `fonts-dejavu-core` |
| Date | 2026-09-24 |

## Installing the requirements

`requirements.txt` installs unmodified. The Apple-only packages are skipped by
their `sys_platform` markers, numpy is pinned below 2, and the unrelated
`Image` package is gone ([#6](https://github.com/Bissbert/GemDiagramAI/issues/6) and [#9](https://github.com/Bissbert/GemDiagramAI/issues/9)):

```
exit=0
Django, Image, tensorflow-macos, tensorflow-metal installed: 0
tensorflow 2.14.0 | numpy 1.26.4 | Pillow 12.3.0 | CairoSVG 2.9.1 | matplotlib 3.11.2
```

Before those fixes, pip failed with `Could not find a version that satisfies
the requirement tensorflow-macos`. With those two lines removed, numpy 2.4.6
was installed and `import tensorflow` failed with
`AttributeError: _ARRAY_API not found`.

## Tests

```
23 passed, 4 warnings in 11.98s
```

The suite in [`tests/`](../tests) uses 16 × 16 models and synthetic or fixture
data. [`tests/docker.sh`](../tests/docker.sh) runs it on its own. For every
fixed issue, reverting the fix makes its tests fail:

| Reverted | Failing tests |
|---|---|
| `model.py` to before [#4](https://github.com/Bissbert/GemDiagramAI/issues/4) | 9, including all of `tests/test_model.py` and the end-to-end run |
| `histogram_freq=0` back to `1` ([#7](https://github.com/Bissbert/GemDiagramAI/issues/7)) | `test_train_disables_tensorboard_histograms` |
| `prepare_data_for_generation.py` to before [#5](https://github.com/Bissbert/GemDiagramAI/issues/5) | 3 in `tests/test_pipeline.py` |
| `requirements.txt` to before [#6](https://github.com/Bissbert/GemDiagramAI/issues/6) and [#9](https://github.com/Bissbert/GemDiagramAI/issues/9) | 5 in `tests/test_requirements.py` |
| blanks no longer imputed ([#8](https://github.com/Bissbert/GemDiagramAI/issues/8)) | 2 in `tests/test_data_utils.py` |

## Configuration

`tools/measure_config.py` prints the values in `constants.py`:

```
IMAGE_SIZE = 512
EPOCHS = 2000
BATCH_SIZE = 32
SAVE_INTERVAL = 20
DEFAULT_GENERATION_DATA_DIR = generation-data
DEFAULT_TRAINING_DATA_DIR = training-data
```

## The prepared dataset

`tools/measure_dataset.py` reads the NPZ headers without expanding the image
tensor:

```
shape       : (4992, 512, 512, 3)  dtype=float32
on disk     : 474,755,860 bytes
in memory   : 15,703,474,304 bytes (14.63 GiB)
per diagram : 3,145,728 bytes
```

The metadata has **13** fields. **3** are numeric and standardized
(`faceCount`, `lengthWidthRatio`, `volumeWidthCubedRatio`); the other 10 are
stored as text. `indexWheel` and `symmetry` hold identical values.

This copy predates the fix for [#8](https://github.com/Bissbert/GemDiagramAI/issues/8), so it has no `metadata_stats.json`:

```
conditioning  : no metadata_stats.json (prepared before it existed)
```

Five of its text fields hold only numbers and blanks. Preparing the source
SVGs again parses them as numeric, which gives 8 numeric fields. The source
SVGs are not part of this run, so the 3-field copy is what is measured here.

## Preparing the fixtures

The three fixture SVGs in `tools/fixtures/` were prepared with NumPy runtime
warnings promoted to errors:

```
exit=0 (RuntimeWarning promoted to error)
npz files: 14
stats file: metadata_stats.json
```

**8** of the 13 fields parse as numeric, and all 8 are used for conditioning:

```
conditioning  : (3, 8) dtype=float64 from metadata_stats.json
```

All three share one `lengthWidthRatio`, which now normalizes to zeros instead of `NaN`:

```
lengthWidthRatio         float64   yes              0       1  mean=+0.0000 std=0.0000 min=+0.000 max=+0.000
```

That is the fix from
[`08be6ff`](https://github.com/Bissbert/GemDiagramAI/commit/08be6ff).

## The model

`tools/measure_model.py` builds the three conditioned Keras models with 8
metadata fields:

```
=== after_build_combined ===
  generator      inputs=[[None, 100], [None, 8]] total=228,813,443 trainable=228,813,059
  discriminator  inputs=[[None, 512, 512, 3], [None, 8]] total=1,471,817 trainable=0
  combined       inputs=[[None, 100], [None, 8]] total=230,285,260 trainable=228,813,059
```

The generator's first layer, `Dense [None, 2097152]` after the 108-wide
noise-plus-metadata concatenation, has **228,589,568** of those parameters.

## Training wiring

`tools/check_training_wiring.py --batch-size 1` fits each model once on one
example and compares weights before and after:

```
control (never combined)           trainable_params=1,470,921  max_weight_delta=1.886e-01  learned=yes
subject (after build_combined)     trainable_params=        0  max_weight_delta=1.890e-01  learned=yes
combined.fit -> discriminator      max_weight_delta=0.000e+00  frozen=yes
combined.fit -> generator          max_weight_delta=2.000e-01  learned=yes
```

The discriminator learns in its own `fit` and stays frozen inside the combined
model. That is the intended setup.

## Inference

`tools/check_inference_path.py` imports what `run_model.py` imports and calls
the conditioned generator both ways:

```
matplotlib.pyplot                            installed
requirements.txt lists matplotlib: True
generator inputs                             [[None, 100], [None, 8]]
predict([z, metadata])                       ok     (1, 512, 512, 3)
predict(z), metadata omitted                 FAILS  ValueError: ... Layer "generator" expects 2 input(s), but it received 1 input tensors.
```

`prepare_data_for_generation.py --save_dir <new dir> --training_data_dir <fixture output>`,
run from an empty working directory. The eight answers are
`65, 16, 1.0, blank, blank, 0.5, 0.17, 0.2`:

```
exit=0
working directory now holds: gem_cutting_diagrams.log output
output/ holds: generation_metadata.npz
arr_0 (1, 8) [[-0.226, -0.707, 0.0, 0.0, 0.0, 1.5, 0.542, -0.903]]
```

The two blank answers and the constant `lengthWidthRatio` normalize to 0. The
log file is the scripts' usual `gem_cutting_diagrams.log` in the working
directory.

## One training epoch

`tools/measure_training_step.py --epochs 1 --batch-size 1 --images random` runs
`model.train` for one epoch at full size on random images, with 8 random
metadata fields:

```
0 [D loss: 0.936586, acc.: 0.00%] [G loss: 0.920787]

total       : 5.06 s for 1 epochs
per epoch   : 5.06 s
projected   : 2.81 h for constants.EPOCHS = 2000
checkpoint  : generator_model_epoch_0.h5 915,291,768 bytes (872.9 MiB)
exit=0 wall=6s
```

Before the fix for [#7](https://github.com/Bissbert/GemDiagramAI/issues/7) this failed after 6 s:
`OOM when allocating tensor with shape[209715200,30] and type double ... [Op:OneHot]`.
The projection is for a batch of 1, not the default 32.

## Media

| File | Bytes | Made by |
|---|---:|---|
| `media/dataset-samples.png` | 394,551 | `tools/render_samples.py`: 12 of the 4,992 prepared diagrams |
| `media/dataset-by-girdles.png` | 279,720 | `tools/render_samples.py`: the first diagram at each common girdle count |
| `media/preprocessing-stages.png` | 87,755 | `tools/render_preprocessing.py` on `tools/fixtures/svg/pc01001.svg` |
| `media/parameter-grid.png` | 87,830 | `tools/render_parameter_grid.py`: the same fixture at 128, 256 and 512 px |

`render_preprocessing.py` printed:

```
native   : RGB (690, 460)
resized  : RGB (512, 512)
tensor   : float32 (512, 512, 3) min=-1.0000 max=+1.0000 mean=+0.9587
```

All four images show source diagrams and preprocessing. None is output from a
trained generator.

## Not measured

- Training time at the default batch size, and the quality of a trained
  generator. No full training run was done, and no trained checkpoint is
  committed.
- Whether the metadata controls the generated diagram in a useful way. The
  tests show that the metadata changes the output of an untrained model. They
  say nothing about a trained one.
- The macOS `tensorflow-metal` path. The target here is Linux.
