#!/usr/bin/env python3
"""Print the runtime configuration values used by the project.

This is deliberately small so documentation can quote the values printed by a
real command instead of copying them from source by hand.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import constants


def main():
    for name in (
        "IMAGE_SIZE",
        "EPOCHS",
        "BATCH_SIZE",
        "SAVE_INTERVAL",
        "DEFAULT_GENERATION_DATA_DIR",
        "DEFAULT_TRAINING_DATA_DIR",
    ):
        print(f"{name} = {getattr(constants, name)}")


if __name__ == "__main__":
    main()
