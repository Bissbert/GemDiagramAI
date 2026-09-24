# How this was measured

[← back to the overview](../README.md)

Every number and raster image in this documentation comes from one script run
in a Linux container:

```sh
sh tools/linux-run.sh training-data > media/captures/linux-run.txt
```

[`tools/linux-run.sh`](../tools/linux-run.sh) starts `python:3.11-slim-bookworm`,
mounts the repository and the prepared dataset read-only, copies the repository,
installs the requirements, and runs each script in [`tools/`](../tools). The
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

`requirements.txt` does not install on Linux as it stands:

```
unmodified requirements.txt: exit=1
ERROR: Could not find a version that satisfies the requirement tensorflow-macos (from versions: none)
```

The script drops the `tensorflow-macos` and `tensorflow-metal` lines. The
unpinned `numpy` then resolves to numpy 2, which TensorFlow 2.14 cannot import:

```
numpy 2.4.6
import tensorflow: exit=1
AttributeError: _ARRAY_API not found
```

Reinstalling with the constraint `numpy<2` works:

```
exit=0
tensorflow 2.14.0 | numpy 1.26.4 | Pillow 12.3.0 | CairoSVG 2.9.1 | matplotlib 3.11.2
```

Both problems are entry 6 in [Bugs found](BUGS-FOUND.md).

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

## Preparing the fixtures

The three fixture SVGs in `tools/fixtures/` were prepared with NumPy runtime
warnings promoted to errors:

```
exit=0 (RuntimeWarning promoted to error)
npz files: 14
```

With only three rows, **8** of the 13 fields parse as numeric. All three share
one `lengthWidthRatio`, which now normalizes to zeros instead of `NaN`:

```
lengthWidthRatio         float64   yes              0       1  mean=+0.0000 std=0.0000 min=+0.000 max=+0.000
```

That is the fix from entry 1 in [Bugs found](BUGS-FOUND.md).

## The model

`tools/measure_model.py` builds the three Keras models:

```
=== after_build_combined ===
  generator      inputs=[[None, 100]] total=212,036,227 trainable=212,035,843
  discriminator  inputs=[[None, 512, 512, 3]] total=1,471,809 trainable=0
  combined       inputs=[[None, 100]] total=213,508,036 trainable=212,035,843
```

The generator's first layer, `Dense [None, 2097152]`, has **211,812,352** of
those parameters.

## Training wiring

`tools/check_training_wiring.py --batch-size 1` fits each model once on one
example and compares weights before and after:

```
control (never combined)           trainable_params=1,470,913  max_weight_delta=1.896e-01  learned=yes
subject (after build_combined)     trainable_params=        0  max_weight_delta=2.018e-01  learned=yes
combined.fit -> discriminator      max_weight_delta=0.000e+00  frozen=yes
combined.fit -> generator          max_weight_delta=2.000e-01  learned=yes
```

The discriminator learns in its own `fit` and stays frozen inside the combined
model. That is the intended setup; see entry 3 in [Bugs found](BUGS-FOUND.md).

## Inference

`tools/check_inference_path.py` imports what `run_model.py` imports and calls
the generator both ways:

```
matplotlib.pyplot                            installed
requirements.txt lists matplotlib: True
generator inputs                             [[None, 100]]
predict(z)                                   ok     (1, 512, 512, 3)
predict([z, combined_metadata])              FAILS  ValueError: Layer "sequential" expects 1 input(s), but it received 2 input tensors.
```

`prepare_data_for_generation.py --save_dir <new dir>`, run from an empty
working directory:

```
exit=1
FileNotFoundError: [Errno 2] No such file or directory: 'training_data_meta_meta1.npz'
working directory now holds: gem_cutting_diagrams.log
```

## One training epoch

`tools/measure_training_step.py --epochs 1 --batch-size 1 --images random` runs
`model.train` for one epoch on random images:

```
1/1 [==============================] - 1s 665ms/step
...
tensorflow.python.framework.errors_impl.ResourceExhaustedError: ... OOM when allocating tensor with shape[209715200,30] and type double ... [Op:OneHot]
exit=1 wall=6s
```

The discriminator steps finish. The first `combined.fit` then fails inside the
TensorBoard histogram callback, and no checkpoint is written. This is entry 7 in
[Bugs found](BUGS-FOUND.md). Because the loop cannot finish one epoch, no
training time is reported.

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

- Training time, and the quality of a trained generator. Training stops in the
  first epoch (entry 7), and no trained checkpoint is committed.
- Metadata-conditioned generation. The generator takes one input (entry 2).
- The macOS `tensorflow-metal` path. The target here is Linux.
