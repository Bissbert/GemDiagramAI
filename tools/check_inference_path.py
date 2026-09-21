#!/usr/bin/env python3
"""Check the inference path that run_model.py takes, step by step.

run_model.py does three things this script reproduces without needing a
trained checkpoint or a display:

  1. imports matplotlib.pyplot at module scope,
  2. column_stacks every .npz in the generation directory,
  3. calls generator.predict([z, combined_metadata]).

Step 3 is the interesting one: model.build_generator() returns a Sequential
with a single (None, 100) input, so a two-element list is not a valid call.
This script builds that generator and reports what each step actually does.

Usage:
    python tools/check_inference_path.py
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def check(label, fn):
    try:
        result = fn()
    except BaseException as exc:  # noqa: BLE001 - reporting, not handling
        # Keras wraps the real cause at the end of a long traceback string.
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        cause = lines[-1] if lines else ""
        print(f"  {label:44s} FAILS  {type(exc).__name__}: {cause[:140]}")
        return None
    print(f"  {label:44s} ok     {result}")
    return result


def main():
    import numpy as np

    print("step 1: module-scope imports of run_model.py")
    for module in ("matplotlib.pyplot", "tensorflow", "numpy"):
        found = importlib.util.find_spec(module.split(".")[0]) is not None
        state = "installed" if found else "NOT INSTALLED"
        print(f"  {module:44s} {state}")
    print("  requirements.txt lists matplotlib:",
          "matplotlib" in open("requirements.txt").read())
    print()

    from model import build_generator, z_dim

    generator = build_generator(z_dim)
    print("step 3: generator.predict signature")
    print(f"  generator inputs                             "
          f"{[list(t.shape) for t in generator.inputs]}")

    z = np.random.normal(0, 1, (1, z_dim)).astype("float32")
    metadata = np.zeros((1, 13), dtype="float32")

    check("predict(z)                       ",
          lambda: generator.predict(z, verbose=0).shape)
    check("predict([z, combined_metadata])  ",
          lambda: generator.predict([z, metadata], verbose=0).shape)


if __name__ == "__main__":
    main()
