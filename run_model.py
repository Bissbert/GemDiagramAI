import os
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from constants import DEFAULT_GENERATION_DATA_DIR


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gem_cutting_diagrams.log", "a"),
                              logging.StreamHandler()])


def main():
    # Move argparse below
    parser = argparse.ArgumentParser(description="Run the GAN model.")
    parser.add_argument('--generation_data_dir', default=DEFAULT_GENERATION_DATA_DIR, help='Directory where generation data is saved.')
    args = parser.parse_args()
    # Load the trained generator model
    model_path = input("Enter the path to the trained generator model (e.g., generator_model_final.h5): ").strip()
    generator = load_model(model_path)

    # Generate a diagram using random noise and loaded normalized metadata
    z = np.random.normal(0, 1, (1, 100))

    # Load normalized metadata from .npz file

    # Dynamically load all .npz metadata files excluding the image file
    meta_files = [f for f in os.listdir(args.generation_data_dir) if f.endswith('.npz') and "training_data_imgs" not in f]
    metadata_arrays = [np.load(os.path.join(args.generation_data_dir, f))['arr_0'] for f in meta_files]

    # Combine the loaded normalized metadata values
    combined_metadata = np.column_stack(metadata_arrays)

    generated_image = generator.predict([z, combined_metadata])

    # Rescale the generated image to the range [0, 255]
    generated_image = 0.5 * generated_image + 0.5

    # Display the generated image
    plt.imshow(generated_image[0])
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
