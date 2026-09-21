# Bugs found during the documentation pass

[← back to the overview](../README.md)

This pass did not change the tracked implementation. The entries below record
behaviour reproduced with the commands shown, followed by the fix that would
normally be made.

## Metadata normalization divides by zero

**File and line:** `data_utils.py:40-45`

**What happens:** numeric metadata with zero standard deviation is divided by
zero and becomes `NaN`. The three-fixture quick start has a constant
`lengthWidthRatio` column, so the preparation run emits a NumPy runtime warning
and the measurement script reports that field as all `NaN`.

**Reproduce:**

```sh
quickstart_dir=$(mktemp -d /tmp/gemdiagram-bug.XXXXXX)
printf '%s\n%s\n' "$PWD/tools/fixtures/svg" "$PWD/tools/fixtures/metadata.json" |
  tf_m1_env/bin/python prepare_data_for_training.py --save_dir "$quickstart_dir"
tf_m1_env/bin/python tools/measure_dataset.py --data-dir "$quickstart_dir"
```

**Fix I would have made:** preserve a zero-valued normalized column, or reject
it explicitly, instead of dividing by zero.

```diff
diff --git a/data_utils.py b/data_utils.py
@@
-    return (metadata_array - metadata_array.mean(axis=0)) / metadata_array.std(axis=0)
+    mean = metadata_array.mean(axis=0)
+    std = metadata_array.std(axis=0)
+    return np.divide(metadata_array - mean, std, out=np.zeros_like(metadata_array),
+                     where=std != 0)
```

## Metadata is collected but never conditions the generator

**File and line:** `model.py:26-49`, `model.py:100-124`, and
`train_model.py:28-44`

**What happens:** the generator has one input, the 100-value noise vector.
`train_model.py` loads and column-stacks metadata, but `model.train` never reads
its `metadata` argument. The repository therefore does not currently implement
the conditional input described by the original README.

**Reproduce:**

```sh
tf_m1_env/bin/python tools/check_inference_path.py
```

The command reports `generator inputs [[None, 100]]`; `predict(z)` succeeds and
`predict([z, combined_metadata])` fails because the model expects one input.

**Fix I would have made:** define a second metadata input, combine it with the
noise path in the generator, and pass the same two-input signature through
training and inference.

```diff
diff --git a/model.py b/model.py
@@
-def build_generator(z_dim):
+def build_generator(z_dim, metadata_dim):
+    noise = Input(shape=(z_dim,))
+    metadata = Input(shape=(metadata_dim,))
+    inputs = Concatenate()([noise, metadata])
@@
-    return model
+    return Model([noise, metadata], model(inputs))
```

## The discriminator is marked frozen after it was compiled

**File and line:** `model.py:51-63`, `model.py:94-96`, and
`model.py:126-128`

**What happens:** `build_combined` sets `discriminator.trainable = False`, but
the discriminator was already compiled. The training-wiring measurement found
zero current trainable parameters after combining while a direct
`discriminator.fit` still changed weights.

**Reproduce:**

```sh
tf_m1_env/bin/python tools/check_training_wiring.py --batch-size 1
```

The run reports `trainable_params=0` with `learned=yes` for the subject after
`build_combined`, while `combined.fit` correctly reports the discriminator as
frozen.

**Fix I would have made:** keep separately compiled discriminator and combined
training views, or recompile the discriminator after changing its trainable
state before calling `fit`.

```diff
diff --git a/model.py b/model.py
@@
-    discriminator.trainable = False
+    discriminator.trainable = True
+    discriminator.compile(
+        loss='binary_crossentropy',
+        optimizer=tf.keras.optimizers.legacy.Adam(0.0002, 0.5),
+        metrics=['accuracy'])
+    discriminator.trainable = False
    model = Sequential([generator, discriminator])
```

## Generation preparation does not write to its requested directory

**File and line:** `prepare_data_for_generation.py:25`,
`prepare_data_for_generation.py:32-36`

**What happens:** `--save_dir` is parsed but ignored. The script then looks for
the hard-coded files `training_data_meta_meta1.npz` and
`training_data_meta_meta2.npz`, and writes `generation_metadata.npz` in the
current working directory. In a clean directory the first lookup fails before
the output is written.

**Reproduce:**

```sh
repo_dir=$PWD
failure_dir=$(mktemp -d /tmp/gemdiagram-generation-failure.XXXXXX)
cd "$failure_dir"
printf '0.5\n0.8\n' |
  PYTHONPATH="$repo_dir" "$repo_dir/tf_m1_env/bin/python" \
  "$repo_dir/prepare_data_for_generation.py" --save_dir "$failure_dir/output"
```

The command raises `FileNotFoundError` for
`training_data_meta_meta1.npz`. If those files existed, the output call still
uses the current directory and saves named keys (`meta1`, `meta2`) rather than
the `arr_0` key expected by the training-data loader pattern.

**Fix I would have made:** resolve the stats files and output path under
`args.save_dir`, and use one documented NPZ key convention end to end.

```diff
diff --git a/prepare_data_for_generation.py b/prepare_data_for_generation.py
@@
-    meta1_normalized = normalize_metadata_using_training_stats(meta1, "training_data_meta_meta1.npz")
+    meta1_normalized = normalize_metadata_using_training_stats(
+        meta1, os.path.join(args.save_dir, "training_data_meta_meta1.npz"))
@@
-    np.savez_compressed("generation_metadata.npz", meta1=meta1_normalized, meta2=meta2_normalized)
+    os.makedirs(args.save_dir, exist_ok=True)
+    np.savez_compressed(os.path.join(args.save_dir, "generation_metadata.npz"),
+                        arr_0=np.column_stack([meta1_normalized, meta2_normalized]))
```

## Inference also depends on an undeclared plotting dependency

**File and line:** `run_model.py:5`

**What happens:** `run_model.py` imports `matplotlib.pyplot`, but
`requirements.txt` does not list `matplotlib`. The inference-path check found
that module unavailable in the project environment before the model could be
run. The same check independently reaches the two-input mismatch above.

**Reproduce:**

```sh
tf_m1_env/bin/python tools/check_inference_path.py
```

The output reports `matplotlib.pyplot NOT INSTALLED` and
`requirements.txt lists matplotlib: False`.

**Fix I would have made:** either declare the plotting dependency or move the
plot import behind an optional output path with a clear error message.

```diff
diff --git a/requirements.txt b/requirements.txt
@@
 numpy
+matplotlib
 tensorflow-macos
```
