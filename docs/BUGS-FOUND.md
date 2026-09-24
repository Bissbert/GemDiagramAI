# Bugs found

[← back to the overview](../README.md)

Each entry below was reproduced with the command shown and reviewed. Eight
were fixed and one was rejected as intended behaviour. Entries 6 to 9 turned up
when everything was re-run in a Linux container (see
[How this was measured](measurement.md)). The regression tests are in
[`tests/`](../tests), and [`tests/docker.sh`](../tests/docker.sh) runs them in
Linux.

| # | Entry | Status |
|---|---|---|
| 1 | Metadata normalization divides by zero | Fixed in [`08be6ff`](https://github.com/Bissbert/GemDiagramAI/commit/08be6ff) |
| 2 | Metadata never conditions the generator | Fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#4](https://github.com/Bissbert/GemDiagramAI/issues/4)) |
| 3 | Discriminator compiled before being frozen | Not a bug: intended GAN wiring |
| 4 | Generation preparation ignores `--save_dir` | Fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#5](https://github.com/Bissbert/GemDiagramAI/issues/5)) |
| 5 | `matplotlib` not declared | Fixed in [`438f225`](https://github.com/Bissbert/GemDiagramAI/commit/438f225) |
| 6 | `requirements.txt` does not install on Linux | Fixed in [`45dc14a`](https://github.com/Bissbert/GemDiagramAI/commit/45dc14a) ([#6](https://github.com/Bissbert/GemDiagramAI/issues/6)) |
| 7 | TensorBoard histograms exhaust memory in the first epoch | Fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#7](https://github.com/Bissbert/GemDiagramAI/issues/7)) |
| 8 | Metadata fields with a blank are stored as text | Fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#8](https://github.com/Bissbert/GemDiagramAI/issues/8)) |
| 9 | `requirements.txt` installs the unrelated `Image` package | Fixed in [`45dc14a`](https://github.com/Bissbert/GemDiagramAI/commit/45dc14a) ([#9](https://github.com/Bissbert/GemDiagramAI/issues/9)) |

The commands below run inside the container that
[`tools/linux-run.sh`](../tools/linux-run.sh) sets up, from a copy of the
repository.

## 1. Metadata normalization divides by zero

**Status:** fixed in [`08be6ff`](https://github.com/Bissbert/GemDiagramAI/commit/08be6ff).

**File:** `data_utils.py:40-53` (`normalize_metadata`)

**What happened:** numeric metadata with zero standard deviation was divided by
zero and became `NaN`. The three fixture SVGs share one `lengthWidthRatio`, so
preparing them emitted a NumPy `RuntimeWarning` and stored that field as all
`NaN`.

**What changed:** the mean and standard deviation are computed once and the
division uses `np.divide(..., out=zeros, where=std != 0)`. A constant column now
normalizes to finite zeros.

**Check:**

```sh
mkdir /tmp/fx
printf '%s\n%s\n' "$PWD/tools/fixtures/svg" "$PWD/tools/fixtures/metadata.json" |
  python3 -W error::RuntimeWarning prepare_data_for_training.py --save_dir /tmp/fx
python3 tools/measure_dataset.py --data-dir /tmp/fx
```

With warnings promoted to errors the preparation exits 0 and writes 14 NPZ
files. `lengthWidthRatio` is reported as `std=0.0000 min=+0.000 max=+0.000`.

## 2. Metadata never conditions the generator

**Status:** fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#4](https://github.com/Bissbert/GemDiagramAI/issues/4)).

**Files:** `model.py`, `train_model.py`, `run_model.py`

**What happened:** the generator had one input, the 100-value noise vector.
`train_model.py` loaded and column-stacked the metadata, but `model.train`
never read its `metadata` argument. `run_model.py` then called the generator
with noise and metadata, which Keras rejected with
`Layer "sequential" expects 1 input(s), but it received 2 input tensors`.

**What changed:** the GAN is now conditional.

- `build_generator(z_dim, meta_dim, image_size)` takes `[z, metadata]` and
  concatenates the two before the first `Dense` layer.
- `build_discriminator(img_shape, meta_dim)` takes `[image, metadata]` and
  concatenates the metadata with the flattened features before the output
  layer.
- `build_combined` passes the same metadata to both.
- `model.train` samples a metadata row for every image and passes it to every
  `predict` and `fit` call.
- `train_model.py` conditions on the numeric fields listed in
  `metadata_stats.json` (see entry 4), in that order.

With `meta_dim=0` the models keep their original single-input form.

**Check:**

```sh
python3 tools/check_inference_path.py
```

```
generator inputs                             [[None, 100], [None, 8]]
predict([z, metadata])                       ok     (1, 512, 512, 3)
predict(z), metadata omitted                 FAILS  ValueError: ... Layer "generator" expects 2 input(s), but it received 1 input tensors.
```

Regression tests in `tests/test_model.py` check the following:

- the generator's and discriminator's outputs change with the metadata;
- `model.train` passes rows of the metadata to all three models.

`tests/test_pipeline.py::test_end_to_end_conditioned_generation` goes from the
fixture SVGs to a generated PNG.

## 3. Discriminator compiled before being frozen

**Status:** not a bug. This entry was reviewed and rejected; nothing was
changed.

**Files:** `model.py` (`build_combined`, `build_discriminator`)

**What was reported:** `build_combined` sets `discriminator.trainable = False`
after the discriminator was compiled, and the standalone `discriminator.fit`
still changes weights afterwards.

**Why that is intended:** this is the usual Keras GAN setup. The discriminator
is compiled while trainable, so its own `fit` calls learn. The combined model
is compiled after the flag is cleared, so `combined.fit` only updates the
generator. Keras keeps the trainable state each model had when it was compiled.
Recompiling the discriminator while frozen would stop it learning.

The wiring check shows exactly those two phases, now with the conditioned
models. `tests/test_model.py::test_discriminator_learns_and_combined_freezes_it`
keeps this behaviour in place:

```sh
python3 tools/check_training_wiring.py --batch-size 1
```

```
control (never combined)           trainable_params=1,470,921  max_weight_delta=1.886e-01  learned=yes
subject (after build_combined)     trainable_params=        0  max_weight_delta=1.890e-01  learned=yes
combined.fit -> discriminator      max_weight_delta=0.000e+00  frozen=yes
combined.fit -> generator          max_weight_delta=2.000e-01  learned=yes
```

## 4. Generation preparation ignores `--save_dir`

**Status:** fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#5](https://github.com/Bissbert/GemDiagramAI/issues/5)).

**Files:** `prepare_data_for_training.py`, `prepare_data_for_generation.py`,
`run_model.py`

**What happened:** `--save_dir` was parsed and then never used. The script
read the hard-coded relative files `training_data_meta_meta1.npz` and
`training_data_meta_meta2.npz`, which nothing writes, and exited with
`FileNotFoundError`. Had it got further, it would have written
`generation_metadata.npz` to the current directory with the keys `meta1` and
`meta2`, while `run_model.py` read `arr_0` from every NPZ in
`--generation_data_dir`.

**What changed:**

- `prepare_data_for_training.py` writes `metadata_stats.json` next to the
  tensors. It lists the numeric fields in order, with the mean and standard
  deviation used to normalize each one.
- `prepare_data_for_generation.py` takes `--training_data_dir` and reads those
  statistics. It prompts once per field (a blank answer means the training
  mean) and normalizes the answers the same way. It creates `--save_dir` and
  writes `generation_metadata.npz` there, with the `(1, k)` array under
  `arr_0`.
- `run_model.py` reads that one file and checks its width against the model's
  metadata input. It also takes `--model` and `--output` so it can run without
  a prompt or a display.

**Check**, from an empty working directory, using the fixture output of entry 1:

```sh
mkdir /tmp/empty && cd /tmp/empty
printf '65\n16\n1.0\n\n\n0.5\n0.17\n0.2\n' |
  PYTHONPATH=/tmp/gd python3 /tmp/gd/prepare_data_for_generation.py \
    --save_dir /tmp/empty/output --training_data_dir /tmp/fx
```

```
exit=0
working directory now holds: gem_cutting_diagrams.log output
output/ holds: generation_metadata.npz
arr_0 (1, 8) [[-0.226, -0.707, 0.0, 0.0, 0.0, 1.5, 0.542, -0.903]]
```

`tests/test_pipeline.py::test_generation_prep_writes_into_save_dir` checks the
file location, the normalized values and that nothing is written to the
working directory.

## 5. `matplotlib` not declared

**Status:** fixed in [`438f225`](https://github.com/Bissbert/GemDiagramAI/commit/438f225).

**File:** `requirements.txt`

**What happened:** `run_model.py` imports `matplotlib.pyplot`, but
`requirements.txt` did not list it.

**What changed:** `matplotlib` was added to `requirements.txt`. In the Linux
run `tools/check_inference_path.py` reports:

```
matplotlib.pyplot                            installed
requirements.txt lists matplotlib: True
```

## 6. `requirements.txt` does not install on Linux

**Status:** fixed in [`45dc14a`](https://github.com/Bissbert/GemDiagramAI/commit/45dc14a) ([#6](https://github.com/Bissbert/GemDiagramAI/issues/6)). Found in the Linux run.

**File:** `requirements.txt`

**What happened:** two separate problems.

- `tensorflow-macos` and `tensorflow-metal` have no Linux wheels, so
  `pip install -r requirements.txt` failed before installing anything:
  `Could not find a version that satisfies the requirement tensorflow-macos`.
- With those two lines removed, the unpinned `numpy` resolved to numpy 2
  (2.4.6). TensorFlow 2.14.0 was built against numpy 1 and failed on import
  with `AttributeError: _ARRAY_API not found`.

**What changed:**

- `tensorflow-macos==2.14.0` and `tensorflow-metal==1.1.0` carry the marker
  `sys_platform == "darwin"`.
- numpy is pinned to `numpy>=1.23.5,<2`, the range TensorFlow 2.14 supports.

**Check** in `python:3.11-slim-bookworm`, with `requirements.txt` unmodified:

```
=== pip install -r requirements.txt (unmodified)
exit=0
Django, Image, tensorflow-macos, tensorflow-metal installed: 0
tensorflow 2.14.0 | numpy 1.26.4 | Pillow 12.3.0 | CairoSVG 2.9.1 | matplotlib 3.11.2
```

`tests/test_requirements.py` evaluates the markers for Linux and macOS, checks
the numpy range, and checks that the installed environment imports
TensorFlow 2.14.0 with numpy 1.

## 7. TensorBoard histograms exhaust memory in the first epoch

**Status:** fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#7](https://github.com/Bissbert/GemDiagramAI/issues/7)). Found in the Linux run.

**File:** `model.py` (`train`)

**What happened:** `train` attached
`TensorBoard(log_dir=log_dir, histogram_freq=1, ...)` to every `fit` call.
At the end of the first `combined.fit`, TensorBoard built a histogram of every
weight. The generator's first `Dense` kernel is 100 × 2,097,152 =
209,715,200 values. TensorBoard's bucketing one-hot encodes that into a
`[209715200, 30]` float64 tensor, about 50 GB. The allocation failed even with
one epoch and a batch of one:

```
tensorflow.python.framework.errors_impl.ResourceExhaustedError: ... OOM when allocating tensor with shape[209715200,30] and type double ... [Op:OneHot]
```

**What changed:** `histogram_freq=0`. Losses and the graph are still logged.

**Check**, at full size:

```sh
python3 tools/measure_training_step.py --epochs 1 --batch-size 1 --images random
```

```
0 [D loss: 0.936586, acc.: 0.00%] [G loss: 0.920787]

total       : 5.06 s for 1 epochs
checkpoint  : generator_model_epoch_0.h5 915,291,768 bytes (872.9 MiB)
exit=0 wall=6s
```

`tests/test_model.py::test_train_disables_tensorboard_histograms` records
every `TensorBoard` that `train` creates and checks `histogram_freq == 0`.

## 8. Metadata fields with a blank are stored as text

**Status:** fixed in [`0502bb6`](https://github.com/Bissbert/GemDiagramAI/commit/0502bb6) ([#8](https://github.com/Bissbert/GemDiagramAI/issues/8)). Found while fixing entry 2.

**File:** `data_utils.py` (`separate_metadata`)

**What happened:** `load_data` filled a field missing from an entry with `''`.
One blank turned the whole column into a string array, and
`normalize_metadata` skipped it. In the prepared dataset five fields are
affected, and every non-blank value in them is a number:

| field | blanks |
|---|---|
| `tableWidthRatio` | 497 |
| `culetWidthRatio` | 497 |
| `pavilionWidthRatio` | 170 |
| `crownWidthRatio` | 159 |
| `girdles` | 39 |

Only 3 of the 13 fields came out numeric, so a conditioned model could have
used only those three.

**What changed:** a field whose non-blank values are all numbers is parsed as
float. Blanks (`''`, `null` or a missing key) are imputed with the field mean,
so they are 0 after normalization. Other fields stay text as before.

**Check:** `tests/test_data_utils.py` covers blanks, missing keys, numeric
strings, text fields and constant fields. The prepared dataset in the Linux run
predates this fix, so `tools/measure_dataset.py` still reports it with 3
numeric fields and no `metadata_stats.json`. Re-running
`prepare_data_for_training.py` on the source SVGs gives 8 numeric fields, as
the fixture run does.

## 9. `requirements.txt` installs the unrelated `Image` package

**Status:** fixed in [`45dc14a`](https://github.com/Bissbert/GemDiagramAI/commit/45dc14a) ([#9](https://github.com/Bissbert/GemDiagramAI/issues/9)). Found while fixing entry 6.

**File:** `requirements.txt`

**What happened:** `requirements.txt` listed `Image`. On PyPI that is a Django
application for cropping and thumbnails (1.5.33), not Pillow. Installing it
pulled in Django 5.2 and `six`. The code only uses `from PIL import Image`,
which comes from the `Pillow` line.

**What changed:** the line was removed.
`tests/test_requirements.py::test_no_unrelated_image_package` checks that it
stays out, and the Linux run reports `Django, Image ... installed: 0`.
