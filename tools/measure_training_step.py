#!/usr/bin/env python3
"""Time model.train() for a few epochs and size the checkpoint it writes.

This calls the real training loop, not a reimplementation, so the timing
includes everything model.train() does per epoch: one generator.predict over
the batch, two discriminator.fit calls, one combined.fit call, and the
TensorBoard callback. The models are conditioned on --meta-dim random
metadata fields (8 in the dataset).

model.train() writes a checkpoint whenever ``epoch % save_interval == 0``,
which includes epoch 0. It writes into the current working directory, so this
script chdirs into --workdir (a temporary directory by default) to keep the
repository clean, and reports the checkpoint size it finds there.

Images are read from the prepared tensor when one is available, and are
otherwise random noise of the same shape; --source says which was used.

Usage:
    python tools/measure_training_step.py --epochs 3
    python tools/measure_training_step.py --epochs 3 --images random
"""

import argparse
import glob
import os
import sys
import tempfile
import time
import zipfile

import numpy as np
import numpy.lib.format as npformat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def read_first_images(path, count):
    """Stream the first `count` diagrams out of the prepared tensor."""
    with zipfile.ZipFile(path) as archive:
        with archive.open("arr_0.npy") as handle:
            version = npformat.read_magic(handle)
            shape, _, dtype = npformat._read_array_header(handle, version)
            item = int(np.prod(shape[1:])) * dtype.itemsize
            take = min(count, shape[0])
            raw = handle.read(item * take)
    return np.frombuffer(raw, dtype=dtype).reshape((take,) + tuple(shape[1:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Defaults to constants.BATCH_SIZE.")
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "training-data"))
    parser.add_argument("--images", choices=("prepared", "random"),
                        default="prepared")
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--meta-dim", type=int, default=8)
    args = parser.parse_args()

    import constants
    import model
    from model import build_generator, build_discriminator, build_combined
    from model import img_shape, z_dim

    batch_size = args.batch_size or constants.BATCH_SIZE
    imgs_path = os.path.join(args.data_dir, "training_data_imgs.npz")

    if args.images == "prepared" and os.path.exists(imgs_path):
        imgs = read_first_images(imgs_path, batch_size)
        source = f"prepared ({imgs_path}, first {len(imgs)} diagrams)"
    else:
        rng = np.random.default_rng(0)
        imgs = rng.uniform(-1, 1, (batch_size,) + img_shape).astype("float32")
        source = "random noise"

    print(f"source      : {source}")
    print(f"images      : {imgs.shape} {imgs.dtype}")
    print(f"batch_size  : {batch_size}")
    print(f"epochs      : {args.epochs}")
    print(f"meta_dim    : {args.meta_dim}")

    generator = build_generator(z_dim, args.meta_dim)
    discriminator = build_discriminator(img_shape, args.meta_dim)
    combined = build_combined(generator, discriminator)

    workdir = args.workdir or tempfile.mkdtemp(prefix="gemdiagram-timing-")
    os.makedirs(workdir, exist_ok=True)
    previous = os.getcwd()
    os.chdir(workdir)
    print(f"workdir     : {workdir}")
    print()

    metadata = np.random.default_rng(1).normal(
        0, 1, (len(imgs), args.meta_dim)).astype("float32")

    start = time.perf_counter()
    model.train(generator, discriminator, combined, imgs, metadata,
                epochs=args.epochs, batch_size=batch_size,
                save_interval=max(args.epochs, 1))
    elapsed = time.perf_counter() - start
    os.chdir(previous)

    print()
    print(f"total       : {elapsed:.2f} s for {args.epochs} epochs")
    print(f"per epoch   : {elapsed / args.epochs:.2f} s")
    print(f"projected   : {elapsed / args.epochs * constants.EPOCHS / 3600:.2f} h "
          f"for constants.EPOCHS = {constants.EPOCHS}")

    checkpoints = sorted(glob.glob(os.path.join(workdir, "*.h5")))
    for path in checkpoints:
        print(f"checkpoint  : {os.path.basename(path)} "
              f"{os.path.getsize(path):,} bytes "
              f"({os.path.getsize(path) / 2**20:.1f} MiB)")
    if not checkpoints:
        print("checkpoint  : none written")


if __name__ == "__main__":
    main()
