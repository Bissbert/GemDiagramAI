
# Gem-Cutting Diagram Generator

The Gem-Cutting Diagram Generator is a tool that leverages a Conditional Generative Adversarial Network (cGAN) to produce gem-cutting diagrams based on provided metadata inputs.

## Prerequisites

Before diving in, ensure you have `pip` installed. Follow these steps to set up an environment for TensorFlow on Mac M1/M2:

```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create a virtual environment named 'tf_m1_env' (or a name of your preference)
virtualenv tf_m1_env

# Activate the virtual environment
source tf_m1_env/bin/activate
```

Next, install the necessary libraries:

```bash
pip install -r requirements.txt
```

## File Structure

- `model.py`: Architecture and training code for the cGAN.
- `model_modified.py`: A potential alternative or updated version of the model.
- `data_utils.py`: Utility functions dedicated to data preprocessing.
- `constants.py`: Contains configuration constants.
- `train_model.py`: Script to train the cGAN. Saves the trained generator model.
- `run_model.py`: Script to utilize a pre-trained generator model and generate diagrams based on input metadata.
- `prepare_data_for_training.py`: Prepares training data.
- `prepare_data_for_generation.py`: Prepares generation data.

## Configuration

Any necessary configurations can be done via the `constants.py` file.

## Usage

### 1. Creating Training Data

To prepare training data, execute `prepare_data_for_training.py`. Provide paths to the folder containing SVG images for training and the file containing metadata in JSON format.

Format for the JSON metadata file:

```json
{
	"fileNameWithoutSuffix": {
		"title1": "data",
		"title2": 23
	}
}
```

This execution creates a folder containing .npz files with normalized training data. To save data in a location other than the default (`training-data`), add the `--save_dir` parameter:

```bash
python prepare_data_for_training.py
```

### 2. Training the Model

By default, training data is sourced from the default directory (same as during creation) or as specified using `--training_data_dir`.

Initiate model training with:

```bash
python train_model.py
```

The model saves at intervals of 20 epochs in the command's execution directory. The final model is also preserved as `generator_model_final.h5`.

### 3. Creating Generation Data

To ready data for generating diagrams, run:

```bash
python prepare_data_for_generation.py --save_dir path/to/save_directory
```

The default save directory is `generation-data`.

### 4. Running the Model

To generate diagrams, execute:

```bash
python run_model.py --generation_data_dir path/to/generation_data_directory
```

By default, the script seeks generation data in the `generation-data` directory. During execution, provide the path to the trained generator model (typically `generator_model_final.h5`).

## Future Optimisations

### 1) Use Configuration Files or Command-Line Arguments:
Instead of relying on user input for paths and parameters, use configuration files or command-line arguments. This will make the process less error-prone and more user-friendly.

### 2) Centralize Constants:
Ensure that constants, especially crucial parameters like z_dim, are defined in a single location and imported elsewhere. This will help maintain consistency and reduce potential errors.


### 3) Experiment with Model Architecture:

If the generated images aren't of satisfactory quality, consider experimenting with different architectures, such as:
Using a U-Net based generator.
Incorporating residual connections.
Trying different activation functions or normalization layers.

### 4) Flexible Metadata Input:

Modify the generation script to handle any number of metadata inputs dynamically, rather than expecting a fixed number.


### 5) Enhance Visualization and Output:

For the inference script (run_model.py), consider adding functionality to save generated images in addition to displaying them. This is especially useful for batch generation.
Consider using tools like TensorBoard for better visualization of training metrics, model architecture, and generated samples during training.

### 6) Regular Checkpoints:

During training, periodically save the model's state, not just the generator. This allows for recovery from interruptions and enables experiments using the discriminator if needed.

### 7) Training Data Augmentation:

Depending on the dataset size, consider using data augmentation techniques to artificially increase the size and diversity of your training data, which can help improve the generalization of the model.

### 8) Model Evaluation Metrics:

While GANs are notoriously hard to evaluate, consider using metrics like the Frechet Inception Distance (FID) to quantitatively measure the quality of generated images over time.

### 9) Model Robustness:

To enhance GAN training stability, consider techniques like gradient penalty, spectral normalization, or using different GAN variants (e.g., WGAN, LSGAN).

### 10) Documentation:

Ensure that the codebase is well-documented. While I saw a README.md in the extracted files, ensure it provides a comprehensive overview of the project, setup instructions, and usage guidelines.
