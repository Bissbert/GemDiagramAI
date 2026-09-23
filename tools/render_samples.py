#!/usr/bin/env python3
"""Render real training diagrams out of the prepared .npz tensor.

``prepare_data_for_training.py`` stores every diagram as a float32 tensor in
``training-data/training_data_imgs.npz``, scaled to [-1, 1]. That file is the
exact input the discriminator sees, so decoding it back to PNG is the most
faithful picture of the dataset available without re-rendering the source SVGs.

The image array is read as a stream straight out of the zip member, so the
full 14.6 GiB tensor is never materialised in memory.

Outputs (under --out-dir, default "media"):
    dataset-samples.png     contact sheet of evenly spaced diagrams
    dataset-by-girdles.png  contact sheet grouped by the "girdles" field

Usage:
    python tools/render_samples.py
    python tools/render_samples.py --data-dir training-data --count 12
"""

import argparse
import os
import zipfile

import numpy as np
import numpy.lib.format as npformat
from PIL import Image, ImageDraw, ImageFont

TILE = 240
PAD = 12
LABEL_H = 30
BG = (255, 255, 255)
FG = (36, 41, 47)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    """A real TTF if one is present, otherwise Pillow's bitmap default."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def open_image_stream(path):
    """Yield (header_shape, dtype, file_object) for the arr_0 member."""
    archive = zipfile.ZipFile(path)
    member = "arr_0.npy"
    handle = archive.open(member)
    version = npformat.read_magic(handle)
    shape, fortran, dtype = npformat._read_array_header(handle, version)
    if fortran:
        raise SystemExit("Fortran-ordered array is not supported.")
    return shape, dtype, handle


def read_selected(path, indices):
    """Read only the requested image indices from the stream, in order."""
    shape, dtype, handle = open_image_stream(path)
    item_bytes = int(np.prod(shape[1:])) * dtype.itemsize
    wanted = sorted(set(int(i) for i in indices))
    out = {}
    cursor = 0
    for index in wanted:
        skip = (index - cursor) * item_bytes
        while skip > 0:
            chunk = handle.read(min(skip, 1 << 24))
            if not chunk:
                raise SystemExit(f"Stream ended before index {index}.")
            skip -= len(chunk)
        raw = handle.read(item_bytes)
        if len(raw) != item_bytes:
            raise SystemExit(f"Short read at index {index}.")
        out[index] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
        cursor = index + 1
    handle.close()
    return shape, out


def to_pil(tensor):
    """Undo the (x - 127.5) / 127.5 scaling applied by data_utils.load_data."""
    array = np.clip(tensor * 127.5 + 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def load_meta(data_dir, key):
    path = os.path.join(data_dir, f"training_data_meta_{key}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)["arr_0"]


def contact_sheet(tiles, cols, title):
    """tiles is a list of (PIL image, label)."""
    rows = (len(tiles) + cols - 1) // cols
    width = cols * TILE + (cols + 1) * PAD
    height = rows * (TILE + LABEL_H) + (rows + 1) * PAD + LABEL_H + PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(16)
    label_font = load_font(13)
    draw.text((PAD, PAD), title, fill=FG, font=title_font)

    top0 = PAD + LABEL_H
    for position, (image, label) in enumerate(tiles):
        row, col = divmod(position, cols)
        x = PAD + col * (TILE + PAD)
        y = top0 + PAD + row * (TILE + LABEL_H + PAD)
        sheet.paste(image.resize((TILE, TILE), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline=(208, 215, 222))
        text = label if len(label) <= 34 else label[:33] + "…"
        draw.text((x, y + TILE + 6), text, fill=FG)
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="training-data")
    parser.add_argument("--out-dir", default="media")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    args = parser.parse_args()

    imgs_path = os.path.join(args.data_dir, "training_data_imgs.npz")
    if not os.path.exists(imgs_path):
        raise SystemExit(
            f"{imgs_path} not found. Run prepare_data_for_training.py first."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    names = load_meta(args.data_dir, "name")
    girdles = load_meta(args.data_dir, "girdles")

    shape, _, handle = open_image_stream(imgs_path)
    handle.close()
    total = shape[0]
    print(f"image tensor: {shape} ({total} diagrams)")

    # Sheet 1: evenly spaced across the whole dataset.
    spread = np.linspace(0, total - 1, args.count).round().astype(int)
    _, images = read_selected(imgs_path, spread)
    tiles = []
    for index in spread:
        label = str(names[index]).strip() if names is not None else ""
        tiles.append((to_pil(images[int(index)]), label or f"[unnamed] index {index}"))
    out1 = os.path.join(args.out_dir, "dataset-samples.png")
    contact_sheet(
        tiles, args.cols,
        f"training_data_imgs.npz - {args.count} of {total} diagrams, "
        f"evenly spaced by index"
    ).save(out1)
    print(f"wrote {out1}")

    # Sheet 2: first diagram found for each of the most common girdle counts.
    if girdles is not None:
        values = [str(v) for v in girdles]
        numeric = sorted(
            {v for v in values if v.strip().lstrip("-").isdigit()},
            key=lambda v: int(v),
        )
        picks = []
        for value in numeric[: args.count]:
            picks.append((values.index(value), value))
        if picks:
            _, images2 = read_selected(imgs_path, [i for i, _ in picks])
            tiles2 = [
                (to_pil(images2[i]), f"girdles = {v}") for i, v in picks
            ]
            out2 = os.path.join(args.out_dir, "dataset-by-girdles.png")
            contact_sheet(
                tiles2, args.cols,
                "training_data_imgs.npz - first diagram at each girdle count"
            ).save(out2)
            print(f"wrote {out2}")
    else:
        print("no girdles metadata found; skipped dataset-by-girdles.png")


if __name__ == "__main__":
    main()
