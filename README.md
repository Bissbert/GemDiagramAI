# GemDiagramAI

![GitHub last commit](https://img.shields.io/github/last-commit/Bissbert/GemDiagramAI)

> Trains a conditional GAN on SVG gem-cutting diagrams and metadata, then generates new diagram images from metadata inputs.

## Why

Gem-cutting diagram generation is a niche problem with no off-the-shelf dataset or model. GemDiagramAI provides the full ML pipeline — data preparation, cGAN training on 512x512 images, and inference — so the workflow can be iterated on with real lapidary diagram data. The project is designed for Mac M1/M2 (uses `tensorflow-macos` and `tensorflow-metal`).

## Quick start

```bash
# Set up a virtual environment (required on Apple Silicon)
pip install virtualenv
virtualenv tf_m1_env
source tf_m1_env/bin/activate
pip install -r requirements.txt

# 1. Prepare training data from SVG images + JSON metadata
python prepare_data_for_training.py
# Reads SVGs and a JSON metadata file; writes .npz files to training-data/

# 2. Train the model
python train_model.py
# Saves generator checkpoints every 20 epochs and a final generator_model_final.h5

# 3. Prepare generation inputs
python prepare_data_for_generation.py --save_dir generation-data

# 4. Generate new diagrams
python run_model.py --generation_data_dir generation-data
# Prompts for the path to a trained generator model (e.g. generator_model_final.h5)
```

## How it works

- `model.py` — defines the cGAN architecture. The generator upsamples a 100-dimensional noise vector through two `UpSampling2D + Conv2D` blocks to produce 512x512 RGB images. The discriminator is a four-layer strided-convolution network with LeakyReLU and dropout. Both use Adam (lr=0.0002, beta=0.5).
- `train_model.py` — orchestrates the adversarial training loop for 2000 epochs (configurable in `constants.py`), saving generator checkpoints every 20 epochs.
- `data_utils.py` — preprocessing utilities for loading and normalizing SVG-derived image data.
- `prepare_data_for_training.py` / `prepare_data_for_generation.py` — convert raw SVGs + metadata JSON into `.npz` arrays consumed by the training and inference scripts.
- `run_model.py` — loads a saved generator and produces diagram images from metadata-conditioned inputs.
- `constants.py` — central config: `IMAGE_SIZE=512`, `EPOCHS=2000`, `BATCH_SIZE=32`, `SAVE_INTERVAL=20`.
- Training logs to TensorBoard (`./logs/`).

## Configuration

Edit `constants.py` to change image resolution, epoch count, batch size, or checkpoint frequency. All paths default to `training-data/` and `generation-data/` but can be overridden with CLI arguments.

## Status

Experimental. The model architecture and training pipeline are implemented, but results depend on the quality and quantity of SVG training data supplied by the user. No pre-trained weights are included in the repository.

## License

MIT
