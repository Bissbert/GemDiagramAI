# Bugs found

[← back to the overview](../README.md)

Each entry below was reproduced with the command shown and reviewed. Two were
fixed on `master`. Two are still open because the fix needs a design decision.
One was rejected as intended behaviour. Two more turned up when everything was
re-run in a Linux container (see [How this was measured](measurement.md)).

| # | Entry | Status |
|---|---|---|
| 1 | Metadata normalization divides by zero | Fixed in [`08be6ff`](https://github.com/Bissbert/GemDiagramAI/commit/08be6ff) |
| 2 | Metadata never conditions the generator | Open |
| 3 | Discriminator compiled before being frozen | Not a bug: intended GAN wiring |
| 4 | Generation preparation ignores `--save_dir` | Open |
| 5 | `matplotlib` not declared | Fixed in [`438f225`](https://github.com/Bissbert/GemDiagramAI/commit/438f225) |
| 6 | `requirements.txt` does not install on Linux | Open |
| 7 | TensorBoard histograms exhaust memory in the first epoch | Open |

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

**Status:** open. Conditioning needs decisions about the generator
architecture, the metadata schema and checkpoint compatibility, so no code was
changed.

**Files:** `model.py:26-49`, `model.py:100-160`, `train_model.py:28-44`,
`run_model.py`

**What happens:** the generator has one input, the 100-value noise vector.
`train_model.py` loads and column-stacks the metadata, but `model.train` never
reads its `metadata` argument. `run_model.py` then calls the generator with
noise and metadata, which Keras rejects.

**Reproduce:**

```sh
python3 tools/check_inference_path.py
```

```
generator inputs                             [[None, 100]]
predict(z)                                   ok     (1, 512, 512, 3)
predict([z, combined_metadata])              FAILS  ValueError: Layer "sequential" expects 1 input(s), but it received 2 input tensors.
```

**Possible fix:** give the generator a second metadata input, combine it with
the noise path, and use the same two-input signature in training and
inference.

## 3. Discriminator compiled before being frozen

**Status:** not a bug. This entry was reviewed and rejected; nothing was
changed.

**Files:** `model.py:51-63`, `model.py:94-96`

**What was reported:** `build_combined` sets `discriminator.trainable = False`
after the discriminator was compiled, and the standalone `discriminator.fit`
still changes weights afterwards.

**Why that is intended:** this is the usual Keras GAN setup. The discriminator
is compiled while trainable, so its own `fit` calls learn. The combined model
is compiled after the flag is cleared, so `combined.fit` only updates the
generator. Keras keeps the trainable state each model had when it was compiled.
Recompiling the discriminator while frozen would stop it learning.

The wiring check shows exactly those two phases:

```sh
python3 tools/check_training_wiring.py --batch-size 1
```

```
control (never combined)           trainable_params=1,470,913  max_weight_delta=1.896e-01  learned=yes
subject (after build_combined)     trainable_params=        0  max_weight_delta=2.018e-01  learned=yes
combined.fit -> discriminator      max_weight_delta=0.000e+00  frozen=yes
combined.fit -> generator          max_weight_delta=2.000e-01  learned=yes
```

## 4. Generation preparation ignores `--save_dir`

**Status:** open. The fix needs a decision about where the training statistics
live and which NPZ keys the generation files use, so no code was changed.

**Files:** `prepare_data_for_generation.py:25`, `prepare_data_for_generation.py:32-36`,
`run_model.py:32`

**What happens:** `--save_dir` is parsed and then never used. The script reads
the hard-coded relative files `training_data_meta_meta1.npz` and
`training_data_meta_meta2.npz`, and writes `generation_metadata.npz` to the
current directory with the keys `meta1` and `meta2`. `run_model.py` reads
`arr_0`.

**Reproduce**, from an empty working directory:

```sh
mkdir /tmp/gen && cd /tmp/gen
printf '0.5\n0.8\n' |
  PYTHONPATH=/tmp/gd python3 /tmp/gd/prepare_data_for_generation.py --save_dir /tmp/gen/out
```

```
exit=1
FileNotFoundError: [Errno 2] No such file or directory: 'training_data_meta_meta1.npz'
working directory now holds: gem_cutting_diagrams.log
```

**Possible fix:** take the statistics paths as explicit inputs, write under
`args.save_dir`, and use one NPZ key convention from preparation to inference.

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

**Status:** open. Found in the Linux run.

**File:** `requirements.txt`

**What happens:** two separate problems.

- `tensorflow-macos` and `tensorflow-metal` have no Linux wheels, so
  `pip install -r requirements.txt` fails before installing anything.
- With those two lines removed, the unpinned `numpy` resolves to numpy 2
  (2.4.6 on this run). TensorFlow 2.14.0 was built against numpy 1 and fails
  on import.

**Reproduce** in `python:3.11-slim-bookworm`:

```
unmodified requirements.txt: exit=1
ERROR: Could not find a version that satisfies the requirement tensorflow-macos (from versions: none)
numpy 2.4.6
import tensorflow: exit=1
AttributeError: _ARRAY_API not found
```

Installing with the constraint `numpy<2` gives numpy 1.26.4, and TensorFlow
then imports. `tools/linux-run.sh` installs this way.

**Possible fix:** pin `numpy<2`, and mark the macOS packages with an
environment marker such as `tensorflow-macos; sys_platform == "darwin"`.

## 7. TensorBoard histograms exhaust memory in the first epoch

**Status:** open. Found in the Linux run.

**File:** `model.py:110`

**What happens:** `train` attaches
`TensorBoard(log_dir=log_dir, histogram_freq=1, ...)` to every `fit` call.
At the end of the first `combined.fit`, TensorBoard builds a histogram of every
weight. The generator's first `Dense` kernel is 100 × 2,097,152 =
209,715,200 values. TensorBoard's bucketing one-hot encodes that into a
`[209715200, 30]` float64 tensor, about 50 GB, and the allocation fails. This
happens even with one epoch and a batch of one.

**Reproduce:**

```sh
python3 tools/measure_training_step.py --epochs 1 --batch-size 1 --images random
```

```
1/1 [==============================] - 1s 665ms/step
...
  File "/tmp/gd/model.py", line 144, in train
    g_loss = combined.fit(z, real, epochs=1, verbose=0, callbacks=[tensorboard_callback])
...
tensorflow.python.framework.errors_impl.ResourceExhaustedError: ... OOM when allocating tensor with shape[209715200,30] and type double on /job:localhost/replica:0/task:0/device:CPU:0 by allocator cpu [Op:OneHot] name:
exit=1 wall=6s
```

The container had 33 GB of memory. No checkpoint is written, because the
first save comes after this call.

**Possible fix:** set `histogram_freq=0`, or log histograms only for the small
layers.
