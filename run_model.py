import os
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from constants import DEFAULT_GENERATION_DATA_DIR, GENERATION_METADATA_FILE


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("gem_cutting_diagrams.log", "a"),
                              logging.StreamHandler()])


def main():
    # Move argparse below
    parser = argparse.ArgumentParser(description="Run the GAN model.")
    parser.add_argument('--generation_data_dir', default=DEFAULT_GENERATION_DATA_DIR, help='Directory where generation data is saved.')
    parser.add_argument('--model', help='Trained generator model; prompted for when omitted.')
    parser.add_argument('--output', help='Save the diagram to this PNG instead of displaying it.')
    args = parser.parse_args()
    # Load the trained generator model
    model_path = args.model or input("Enter the path to the trained generator model (e.g., generator_model_final.h5): ").strip()
    generator = load_model(model_path, compile=False)

    # Generate a diagram using random noise and loaded normalized metadata
    z = np.random.normal(0, 1, (1, generator.inputs[0].shape[-1]))

    if len(generator.inputs) > 1:
        # Load the normalized metadata written by prepare_data_for_generation.py
        meta_path = os.path.join(args.generation_data_dir, GENERATION_METADATA_FILE)
        metadata = np.load(meta_path)['arr_0']
        meta_dim = generator.inputs[1].shape[-1]
        if metadata.shape != (1, meta_dim):
            raise SystemExit(f"{meta_path} holds {metadata.shape[-1]} metadata values; "
                             f"the model expects {meta_dim}.")
        generated_image = generator.predict([z, metadata])
    else:
        logging.warning("This generator takes no metadata; generating from noise only.")
        generated_image = generator.predict(z)

    # Rescale the generated image from [-1, 1] to [0, 1]
    generated_image = np.clip(0.5 * generated_image + 0.5, 0, 1)

    if args.output:
        plt.imsave(args.output, generated_image[0])
        logging.info(f"Diagram saved to {args.output}")
        return

    # Display the generated image
    plt.imshow(generated_image[0])
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
