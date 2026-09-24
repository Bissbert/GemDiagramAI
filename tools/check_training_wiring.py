#!/usr/bin/env python3
"""Check whether the discriminator still learns after build_combined().

model.build_combined() sets ``discriminator.trainable = False`` on a
discriminator that was already compiled inside build_discriminator().
model.train() then calls ``discriminator.fit(...)`` on that same object.
This script measures, empirically, whether those fit() calls change any
discriminator weight.

It runs two fit() calls on a tiny random batch and reports the maximum
absolute weight delta, both for a freshly compiled discriminator (control)
and for one that has been through build_combined() (subject).

Usage:
    python tools/check_training_wiring.py [--batch-size 2]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def max_weight_delta(before, after):
    import numpy as np

    return max(
        float(np.abs(a - b).max()) for a, b in zip(after, before) if a.size
    )


def run_case(label, discriminator, batch_size, seed, meta_dim):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (batch_size, 512, 512, 3)).astype("float32")
    if meta_dim:
        x = [x, rng.normal(0, 1, (batch_size, meta_dim)).astype("float32")]
    y = np.ones((batch_size, 1), dtype="float32")

    before = [w.copy() for w in discriminator.get_weights()]
    discriminator.fit(x, y, epochs=1, verbose=0)
    after = discriminator.get_weights()

    trainable = sum(int(w.shape.num_elements())
                    for w in discriminator.trainable_weights)
    delta = max_weight_delta(before, after)
    print(f"{label:34s} trainable_params={trainable:>9,}  "
          f"max_weight_delta={delta:.3e}  learned={'yes' if delta > 0 else 'NO'}")
    return delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--meta-dim", type=int, default=8)
    args = parser.parse_args()

    from model import build_generator, build_discriminator, build_combined
    from model import img_shape, z_dim

    m = args.meta_dim
    control = build_discriminator(img_shape, m)
    run_case("control (never combined)", control, args.batch_size, 0, m)

    generator = build_generator(z_dim, m)
    subject = build_discriminator(img_shape, m)
    combined = build_combined(generator, subject)
    run_case("subject (after build_combined)", subject, args.batch_size, 0, m)

    # Does training the combined model leak updates into the discriminator?
    import numpy as np

    rng = np.random.default_rng(1)
    z = rng.normal(0, 1, (args.batch_size, z_dim)).astype("float32")
    if m:
        z = [z, rng.normal(0, 1, (args.batch_size, m)).astype("float32")]
    y = np.ones((args.batch_size, 1), dtype="float32")

    d_before = [w.copy() for w in subject.get_weights()]
    g_before = [w.copy() for w in generator.get_weights()]
    combined.fit(z, y, epochs=1, verbose=0)
    d_delta = max_weight_delta(d_before, subject.get_weights())
    g_delta = max_weight_delta(g_before, generator.get_weights())
    print(f"{'combined.fit -> discriminator':34s} "
          f"max_weight_delta={d_delta:.3e}  "
          f"frozen={'yes' if d_delta == 0 else 'NO'}")
    print(f"{'combined.fit -> generator':34s} "
          f"max_weight_delta={g_delta:.3e}  "
          f"learned={'yes' if g_delta > 0 else 'NO'}")


if __name__ == "__main__":
    main()
