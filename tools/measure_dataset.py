#!/usr/bin/env python3
"""Measure a prepared training-data directory without loading the image tensor.

Reports, per metadata field: stored dtype, whether normalisation was applied,
the count of blank entries, and summary statistics for the numeric fields. The
image tensor is described from its .npy header only, so this runs in a few
hundred megabytes regardless of dataset size.

Every dataset number quoted in README.md and docs/ comes from this script.

Usage:
    python tools/measure_dataset.py
    python tools/measure_dataset.py --data-dir path/to/training-data
"""

import argparse
import glob
import os
import zipfile

import numpy as np
import numpy.lib.format as npformat

PREFIX = "training_data_meta_"


def header_of(path, member="arr_0.npy"):
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(member)
        with archive.open(member) as handle:
            version = npformat.read_magic(handle)
            shape, fortran, dtype = npformat._read_array_header(handle, version)
    return shape, dtype, info.compress_size, info.file_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="training-data")
    args = parser.parse_args()

    imgs_path = os.path.join(args.data_dir, "training_data_imgs.npz")
    if os.path.exists(imgs_path):
        shape, dtype, packed, unpacked = header_of(imgs_path)
        print("=== image tensor ===")
        print(f"  file        : {imgs_path}")
        print(f"  shape       : {shape}  dtype={dtype}")
        print(f"  on disk     : {packed:,} bytes")
        print(f"  in memory   : {unpacked:,} bytes "
              f"({unpacked / 2**30:.2f} GiB)")
        print(f"  per diagram : {unpacked // shape[0]:,} bytes")
        print()
    else:
        print(f"(no {imgs_path})\n")

    paths = sorted(glob.glob(os.path.join(args.data_dir, PREFIX + "*.npz")))
    if not paths:
        raise SystemExit(f"No {PREFIX}*.npz files in {args.data_dir}.")

    fields = {}
    for path in paths:
        key = os.path.basename(path)[len(PREFIX):-len(".npz")]
        fields[key] = np.load(path, allow_pickle=True)["arr_0"]

    print("=== metadata fields ===")
    header = (f"  {'field':24s} {'dtype':9s} {'normalised':11s} "
              f"{'blank':>6s} {'unique':>7s}  summary")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for key, array in fields.items():
        numeric = np.issubdtype(array.dtype, np.number)
        if numeric:
            blank = int(np.isnan(array).sum())
            unique = len(np.unique(array[~np.isnan(array)]))
            if blank == array.size:
                summary = "all NaN (zero variance in source column)"
            else:
                finite = array[~np.isnan(array)]
                summary = (f"mean={finite.mean():+.4f} std={finite.std():.4f} "
                           f"min={finite.min():+.3f} max={finite.max():+.3f}")
            blank_label = f"{blank}"
        else:
            blank = int((array == "").sum())
            unique = len(np.unique(array))
            summary = "kept as text, not normalised"
            blank_label = f"{blank}"
        print(f"  {key:24s} {str(array.dtype):9s} "
              f"{('yes' if numeric else 'no'):11s} "
              f"{blank_label:>6s} {unique:>7d}  {summary}")

    counts = {k: len(v) for k, v in fields.items()}
    print()
    print(f"  rows          : {sorted(set(counts.values()))}")
    print(f"  fields        : {len(fields)}")
    numeric_fields = [k for k, v in fields.items()
                      if np.issubdtype(v.dtype, np.number)]
    print(f"  numeric fields: {len(numeric_fields)} "
          f"({', '.join(sorted(numeric_fields))})")

    # train_model.py column_stacks every metadata file before training.
    stacked = np.column_stack([fields[k] for k in fields])
    print(f"  column_stack  : {stacked.shape} dtype={stacked.dtype}")

    duplicates = []
    keys = list(fields)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if fields[a].shape == fields[b].shape and np.array_equal(
                fields[a], fields[b]
            ):
                duplicates.append((a, b))
    print(f"  identical pairs: {duplicates if duplicates else 'none'}")


if __name__ == "__main__":
    main()
