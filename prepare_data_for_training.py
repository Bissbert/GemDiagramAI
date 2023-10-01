
import os
import argparse
import json
import numpy as np
import data_utils
import logging
from constants import DEFAULT_TRAINING_DATA_DIR


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gem_cutting_diagrams.log", "a"),
                              logging.StreamHandler()])


def main():
    parser = argparse.ArgumentParser(description='Prepare training data.')
    parser.add_argument('--save_dir', default=DEFAULT_TRAINING_DATA_DIR, help='Directory to save the prepared data.')
    args = parser.parse_args()
    # Prompt user for paths
    svg_folder_path = input("Enter the path to the folder containing the SVG diagrams: ").strip()
    metadata_file_path = input("Enter the path to the metadata JSON file: ").strip()

    # Load metadata from the provided JSON file path
    with open(metadata_file_path, 'r') as f:
        metadata_dict = json.load(f)

    # Load SVG files and their corresponding metadata in order
    svg_files = []
    metadata = []
    for filename, meta_data_value in metadata_dict.items():
        with open(os.path.join(svg_folder_path, filename + ".svg"), 'r') as f:
            svg_files.append(f.read())
            metadata.append(meta_data_value)
        if len(svg_files) % 10 == 0:
            logging.info(f'Loaded {len(svg_files)} SVG files out of {len(metadata_dict)}')

    # Convert SVGs to image tensors and normalize metadata
    imgs, metadata_separated = data_utils.load_data(svg_files, metadata)

    # Save the image tensors and normalized metadata to .npz files

    # Check if the save directory exists and create it if not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.savez_compressed(os.path.join(args.save_dir, "training_data_imgs.npz"), imgs)
    for key, meta_array in metadata_separated.items():
        np.savez_compressed(os.path.join(args.save_dir, f"training_data_meta_{key}.npz"), meta_array)

    logging.info("Training data saved to .npz files.")


if __name__ == "__main__":
    main()
