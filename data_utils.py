import os
import json
import numpy as np
from PIL import Image
from constants import IMAGE_SIZE
import io
import cairosvg
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Written next to the training tensors by prepare_data_for_training.py
METADATA_STATS_FILE = 'metadata_stats.json'


def svg_to_png(svg_str):
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
    return Image.open(io.BytesIO(png_bytes))


def load_data(svg_files, metadata):
    logging.info("Loading and processing SVG files...")
    img_data = [np.array(svg_to_png(file).resize((IMAGE_SIZE, IMAGE_SIZE))) for file in svg_files]
    img_data = np.array(img_data, dtype=np.float32)
    img_data = (img_data - 127.5) / 127.5

    metadata_separated, stats = separate_metadata(metadata)
    return img_data, metadata_separated, stats


def load_metadata_stats(training_data_dir):
    path = os.path.join(training_data_dir, METADATA_STATS_FILE)
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found; run prepare_data_for_training.py "
                         f"to (re)create the training data.")
    with open(path) as f:
        return json.load(f)


def load_training_metadata(training_data_dir, keys):
    """Stack the normalised numeric metadata columns in the given order."""
    columns = [np.load(os.path.join(training_data_dir, f"training_data_meta_{key}.npz"))['arr_0']
               for key in keys]
    return np.column_stack(columns).astype(np.float32)


def is_blank(value):
    return value is None or (isinstance(value, str) and value.strip() == '')


def numeric_column(values):
    """Return values as float64 with blanks as NaN, or None if any non-blank
    value is not a number."""
    parsed = []
    for value in values:
        if is_blank(value):
            parsed.append(np.nan)
            continue
        if isinstance(value, bool):
            return None
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            return None
    column = np.array(parsed, dtype=np.float64)
    if np.isnan(column).all():
        return None
    return column


def separate_metadata(metadata):
    """Split a list of metadata dicts into one array per field.

    A field whose non-blank values are all numbers becomes a normalised float
    column; its blanks are imputed with the field mean (0 after normalising).
    Any other field stays text. Returns (arrays, stats), where stats lists the
    numeric fields in order with the mean and std used to normalise them.
    """
    logging.info("Converting metadata to separate arrays...")
    keys = list(metadata[0].keys())
    metadata_separated = {}
    stats = {"keys": [], "mean": {}, "std": {}}
    for key in keys:
        values = [entry.get(key) for entry in metadata]
        column = numeric_column(values)
        if column is None:
            logging.warning(f'Non-numeric metadata field "{key}"; kept as text.')
            metadata_separated[key] = np.array(['' if is_blank(v) else v for v in values])
            continue
        mean = float(np.nanmean(column))
        column[np.isnan(column)] = mean
        stats["keys"].append(key)
        stats["mean"][key] = mean
        stats["std"][key] = float(column.std())
        metadata_separated[key] = normalize_metadata(column)
    return metadata_separated, stats


def normalize_metadata(metadata_array):
    # Ensure the metadata array contains only numeric values before normalization
    if not np.issubdtype(metadata_array.dtype, np.number):
        logging.warning('Non-numeric metadata detected. Skipping normalization.')
        return metadata_array
    mean = metadata_array.mean(axis=0)
    std = metadata_array.std(axis=0)
    centered = metadata_array - mean
    return np.divide(
        centered,
        std,
        out=np.zeros_like(centered, dtype=np.float64),
        where=std != 0,
    )


def normalize_value(value, mean, std):
    """Normalise one raw metadata value with training-set statistics."""
    return 0.0 if std == 0 else (value - mean) / std
