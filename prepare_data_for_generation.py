import os
import argparse
import logging
import numpy as np
import data_utils
from constants import DEFAULT_GENERATION_DATA_DIR, DEFAULT_TRAINING_DATA_DIR, GENERATION_METADATA_FILE


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gem_cutting_diagrams.log", "a"),
                              logging.StreamHandler()])

def main():
    parser = argparse.ArgumentParser(description="Prepare generation data.")
    parser.add_argument('--save_dir', default=DEFAULT_GENERATION_DATA_DIR, help='Directory to save the prepared data.')
    parser.add_argument('--training_data_dir', default=DEFAULT_TRAINING_DATA_DIR,
                        help='Directory holding the training data and its metadata_stats.json.')
    args = parser.parse_args()

    # Ask for one value per numeric field the model was trained on
    stats = data_utils.load_metadata_stats(args.training_data_dir)
    normalized = []
    for key in stats["keys"]:
        mean, std = stats["mean"][key], stats["std"][key]
        answer = input(f"Enter {key} (blank for the training mean, {mean:.4g}): ").strip()
        value = mean if answer == '' else float(answer)
        normalized.append(data_utils.normalize_value(value, mean, std))

    # Save the normalized metadata values to a .npz file
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, GENERATION_METADATA_FILE)
    np.savez_compressed(save_path, np.array([normalized], dtype=np.float32),
                        keys=np.array(stats["keys"]))

    logging.info(f"Normalized metadata saved for generation to {save_path}")


if __name__ == "__main__":
    main()
