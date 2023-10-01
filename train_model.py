import constants
import argparse
import os
import numpy as np
import model
import logging
from model import build_generator, build_discriminator, build_combined, img_shape
from constants import DEFAULT_TRAINING_DATA_DIR


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gem_cutting_diagrams.log", "a"),
                              logging.StreamHandler()])


def main():
    # Move argparse below
    parser = argparse.ArgumentParser(description="Train the GAN model.")
    parser.add_argument('--training_data_dir', default=DEFAULT_TRAINING_DATA_DIR, help='Directory where training data is saved.')
    args = parser.parse_args()

    # Load image tensors and metadata from .npz files
    imgs = np.load(os.path.join(args.training_data_dir, "training_data_imgs.npz"))['arr_0']

    # For demonstration, we're loading two metadata types. Adjust as needed.

    # Dynamically load all .npz files excluding the image file
    meta_files = [f for f in os.listdir(args.training_data_dir) if f.endswith('.npz') and "training_data_imgs" not in f]
    metadata_arrays = [np.load(os.path.join(args.training_data_dir, f))['arr_0'] for f in meta_files]

    # Combine the loaded metadata arrays
    combined_metadata = np.column_stack(metadata_arrays)

    # Train the model
    epochs = constants.EPOCHS
    batch_size = constants.BATCH_SIZE
    save_interval = constants.SAVE_INTERVAL
    z_dim = 100  # Noise vector size (you might want to adjust this based on your model setup)
    generator = build_generator(z_dim)
    discriminator = build_discriminator(img_shape)
    combined = build_combined(generator, discriminator)

    trained_generator = model.train(generator, discriminator, combined, imgs, combined_metadata, epochs, batch_size, save_interval)

    # Save the final trained generator model
    trained_generator.save("generator_model_final.h5")
    logging.info("Final trained generator model saved to generator_model_final.h5")


if __name__ == "__main__":
    main()
