import argparse
import logging
import numpy as np
from constants import DEFAULT_GENERATION_DATA_DIR


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gem_cutting_diagrams.log", "a"),
                              logging.StreamHandler()])


def normalize_metadata_using_training_stats(metadata_value, training_stats_file):
    # Load training stats (mean and standard deviation)
    training_stats = np.load(training_stats_file)
    mean = training_stats['mean']
    std = training_stats['std']
    return (metadata_value - mean) / std


def main():
    parser = argparse.ArgumentParser(description='Prepare generation data.')
    # Move argparse below
    parser = argparse.ArgumentParser(description="Prepare generation data.")
    parser.add_argument('--save_dir', default=DEFAULT_GENERATION_DATA_DIR, help='Directory to save the prepared data.')
    args = parser.parse_args()
    # For demonstration, we're asking for two metadata inputs. Adjust as needed.
    meta1 = float(input("Enter metadata value 1 (e.g., 0.5): "))
    meta2 = float(input("Enter metadata value 2 (e.g., 0.8): "))

    # Normalize these values using the statistics from the training data
    meta1_normalized = normalize_metadata_using_training_stats(meta1, "training_data_meta_meta1.npz")
    meta2_normalized = normalize_metadata_using_training_stats(meta2, "training_data_meta_meta2.npz")

    # Save the normalized metadata values to a .npz file
    np.savez_compressed("generation_metadata.npz", meta1=meta1_normalized, meta2=meta2_normalized)

    logging.info("Normalized metadata saved for generation.")


if __name__ == "__main__":
    main()
