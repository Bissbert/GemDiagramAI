#!/usr/bin/env python3
"""Measure the GAN architecture defined in model.py.

Reports layer shapes, parameter counts and the input signatures of the
generator and discriminator, before and after ``build_combined`` is called.
Every architecture number quoted in README.md and docs/ comes from this
script.

Usage:
    python tools/measure_model.py [--json]

Run it from the repository root with the project virtualenv active, so that
``import model`` resolves and TensorFlow is importable.
"""

import argparse
import json
import os
import sys

# Allow "python tools/measure_model.py" from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def describe(model):
    """Return a compact description of a Keras model."""
    return {
        "name": model.name,
        "inputs": [list(t.shape) for t in model.inputs],
        "outputs": [list(t.shape) for t in model.outputs],
        "total_params": int(model.count_params()),
        "trainable_params": int(
            sum(int(w.shape.num_elements()) for w in model.trainable_weights)
        ),
        "layers": [
            {
                "name": layer.name,
                "type": type(layer).__name__,
                "output_shape": list(layer.output_shape),
                "params": int(layer.count_params()),
            }
            for layer in model.layers
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    parser.add_argument("--meta-dim", type=int, default=8,
                        help="Numeric metadata fields to condition on "
                             "(8 in the dataset; 0 for an unconditioned GAN).")
    args = parser.parse_args()

    from model import build_generator, build_discriminator, build_combined
    from model import img_shape, z_dim

    generator = build_generator(z_dim, args.meta_dim)
    discriminator = build_discriminator(img_shape, args.meta_dim)

    before = {
        "generator": describe(generator),
        "discriminator": describe(discriminator),
    }

    combined = build_combined(generator, discriminator)

    result = {
        "z_dim": z_dim,
        "meta_dim": args.meta_dim,
        "img_shape": list(img_shape),
        "before_build_combined": before,
        "after_build_combined": {
            "generator": describe(generator),
            "discriminator": describe(discriminator),
            "combined": describe(combined),
        },
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"z_dim      : {z_dim}")
    print(f"meta_dim   : {args.meta_dim}")
    print(f"img_shape  : {img_shape}")
    print()

    for phase in ("before_build_combined", "after_build_combined"):
        print(f"=== {phase} ===")
        for key, desc in result[phase].items():
            print(
                f"  {key:14s} inputs={desc['inputs']} "
                f"total={desc['total_params']:,} "
                f"trainable={desc['trainable_params']:,}"
            )
        print()

    print("=== generator layers ===")
    for layer in before["generator"]["layers"]:
        print(
            f"  {layer['type']:20s} {str(layer['output_shape']):28s} "
            f"{layer['params']:>12,}"
        )
    print()
    print("=== discriminator layers ===")
    for layer in before["discriminator"]["layers"]:
        print(
            f"  {layer['type']:20s} {str(layer['output_shape']):28s} "
            f"{layer['params']:>12,}"
        )


if __name__ == "__main__":
    main()
