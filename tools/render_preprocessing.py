#!/usr/bin/env python3
"""Render the three stages data_utils.load_data puts every SVG through.

Stage 1  cairosvg rasterises the SVG at its intrinsic viewBox size.
Stage 2  Pillow resizes that to IMAGE_SIZE x IMAGE_SIZE, ignoring aspect ratio.
Stage 3  the pixels are rescaled to [-1, 1] by (x - 127.5) / 127.5.

Stage 3 is shown by mapping the signed tensor back to 8-bit for display; the
panel also prints the measured tensor range so the scaling is checkable.

Usage:
    python tools/render_preprocessing.py
    python tools/render_preprocessing.py --svg tools/fixtures/svg/pc01002.svg
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PANEL = 300
PAD = 16
HEAD = 28
CAPTION = 46
BG = (255, 255, 255)
FG = (36, 41, 47)
MUTED = (101, 109, 118)
BORDER = (208, 215, 222)

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def load_font(size):
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def fit(image):
    """Letterbox into a square panel so aspect differences stay visible."""
    canvas = Image.new("RGB", (PANEL, PANEL), (246, 248, 250))
    copy = image.copy()
    copy.thumbnail((PANEL, PANEL), Image.LANCZOS)
    canvas.paste(copy, ((PANEL - copy.width) // 2, (PANEL - copy.height) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", default="tools/fixtures/svg/pc01001.svg")
    parser.add_argument("--out", default="media/preprocessing-stages.png")
    args = parser.parse_args()

    import data_utils
    from constants import IMAGE_SIZE

    with open(args.svg, "r") as handle:
        svg_text = handle.read()

    native = data_utils.svg_to_png(svg_text)
    resized = native.resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = (np.array(resized, dtype=np.float32) - 127.5) / 127.5
    shown = Image.fromarray(
        np.clip(tensor * 127.5 + 127.5, 0, 255).astype(np.uint8)
    )

    stages = [
        (
            fit(native),
            "1. cairosvg.svg2png",
            f"{native.mode} {native.size[0]}x{native.size[1]}, "
            f"aspect {native.size[0] / native.size[1]:.2f}",
        ),
        (
            fit(resized),
            f"2. resize to {IMAGE_SIZE}x{IMAGE_SIZE}",
            f"aspect forced to 1.00, uint8 0..255",
        ),
        (
            fit(shown),
            "3. (x - 127.5) / 127.5",
            f"float32 {tensor.shape}, range "
            f"{tensor.min():+.2f}..{tensor.max():+.2f}",
        ),
    ]

    width = len(stages) * PANEL + (len(stages) + 1) * PAD
    height = HEAD + PANEL + CAPTION + 2 * PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    head_font = load_font(15)
    label_font = load_font(13)
    small_font = load_font(11)

    draw.text(
        (PAD, PAD // 2),
        f"data_utils.load_data on {os.path.basename(args.svg)}",
        fill=FG, font=head_font,
    )

    for position, (image, label, caption) in enumerate(stages):
        x = PAD + position * (PANEL + PAD)
        y = HEAD + PAD // 2
        sheet.paste(image, (x, y))
        draw.rectangle([x, y, x + PANEL - 1, y + PANEL - 1], outline=BORDER)
        draw.text((x, y + PANEL + 8), label, fill=FG, font=label_font)
        draw.text((x, y + PANEL + 26), caption, fill=MUTED, font=small_font)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out)
    print(f"native   : {native.mode} {native.size}")
    print(f"resized  : {resized.mode} {resized.size}")
    print(f"tensor   : {tensor.dtype} {tensor.shape} "
          f"min={tensor.min():+.4f} max={tensor.max():+.4f} "
          f"mean={tensor.mean():+.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
