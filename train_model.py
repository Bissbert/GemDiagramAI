import constants
import argparse
import os
import numpy as np
import model
import logging
import data_utils
from model import build_generator, build_discriminator, build_combined, z_dim
from constants import DEFAULT_TRAINING_DATA_DIR


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gem_cutting_diagrams.log", "a"),
                              logging.StreamHandler()])


def main():
    # Move argparse below
    parser = argparse.ArgumentParser(description="Train the GAN model.")
    parser.add_argument('--training_data_dir', default=DEFAULT_TRAINING_DATA_DIR, help='Directory where training data is saved.')
    parser.add_argument('--epochs', type=int, default=constants.EPOCHS)
    parser.add_argument('--batch_size', type=int, default=constants.BATCH_SIZE)
    parser.add_argument('--save_interval', type=int, default=constants.SAVE_INTERVAL)
    args = parser.parse_args()

    # Load image tensors and metadata from .npz files
    imgs = np.load(os.path.join(args.training_data_dir, "training_data_imgs.npz"))['arr_0']

    # The numeric metadata fields, in the order recorded at preparation time
    stats = data_utils.load_metadata_stats(args.training_data_dir)
    combined_metadata = data_utils.load_training_metadata(args.training_data_dir, stats["keys"])
    meta_dim = combined_metadata.shape[1]
    logging.info(f"Conditioning on {meta_dim} metadata fields: {', '.join(stats['keys'])}")

    # Train the model
    image_size = imgs.shape[1]
    generator = build_generator(z_dim, meta_dim, image_size)
    discriminator = build_discriminator(imgs.shape[1:], meta_dim)
    combined = build_combined(generator, discriminator)

    trained_generator = model.train(generator, discriminator, combined, imgs, combined_metadata,
                                    args.epochs, args.batch_size, args.save_interval)

    # Save the final trained generator model
    trained_generator.save("generator_model_final.h5")
    logging.info("Final trained generator model saved to generator_model_final.h5")


if __name__ == "__main__":
    main()
