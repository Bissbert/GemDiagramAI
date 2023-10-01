import os
import numpy as np
from PIL import Image
from constants import IMAGE_SIZE
import io
import cairosvg
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def svg_to_png(svg_str):
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
    return Image.open(io.BytesIO(png_bytes))


def load_data(svg_files, metadata):
    logging.info("Loading and processing SVG files...")
    img_data = [np.array(svg_to_png(file).resize((IMAGE_SIZE, IMAGE_SIZE))) for file in svg_files]
    img_data = np.array(img_data, dtype=np.float32)
    img_data = (img_data - 127.5) / 127.5

    logging.info("Converting metadata to separate arrays...")
    # Convert metadata list of dictionaries to separate 2D arrays for each metadata type
    # Ensure all metadata entries have expected keys and fill missing keys with blank values
    expected_keys = metadata[0].keys()
    for entry in metadata:
        for key in expected_keys:
            if key not in entry:
                entry[key] = ''  # Fill with blank value
    metadata_separated = {key: np.array([entry[key] for entry in metadata]) for key in metadata[0].keys()}

    # Normalize each metadata type separately
    for key, meta_array in metadata_separated.items():
        metadata_separated[key] = normalize_metadata(meta_array)

    return img_data, metadata_separated


def normalize_metadata(metadata_array):
    # Ensure the metadata array contains only numeric values before normalization
    if not np.issubdtype(metadata_array.dtype, np.number):
        logging.warning('Non-numeric metadata detected. Skipping normalization.')
        return metadata_array
    return (metadata_array - metadata_array.mean(axis=0)) / metadata_array.std(axis=0)
