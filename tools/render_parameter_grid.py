#!/usr/bin/env python3
"""Render one real SVG at several ``IMAGE_SIZE`` values.

The current model does not accept metadata, so a metadata-controlled grid would
be misleading. This grid instead makes the supported raster-size parameter
visible: the same source diagram is rasterised and resized at each requested
size. It uses the project's existing CairoSVG and Pillow dependencies.

Usage:
    python tools/render_parameter_grid.py
    python tools/render_parameter_grid.py --sizes 128 256 512
"""

import argparse
import io
import os

import cairosvg
from PIL import Image, ImageDraw, ImageFont


BG = (255, 255, 255)
FG = (36, 41, 47)
MUTED = (101, 109, 118)
BORDER = (208, 215, 222)
PANEL = 260
PAD = 16
CAPTION = 48


def load_font(size):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def fit(image):
    canvas = Image.new("RGB", (PANEL, PANEL), (246, 248, 250))
    copy = image.convert("RGB")
    copy.thumbnail((PANEL, PANEL), Image.Resampling.LANCZOS)
    canvas.paste(copy, ((PANEL - copy.width) // 2, (PANEL - copy.height) // 2))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--svg", default="tools/fixtures/svg/pc01001.svg",
        help="Source SVG to rasterise.",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[128, 256, 512],
        help="Square raster sizes to compare.",
    )
    parser.add_argument("--out", default="media/parameter-grid.png")
    args = parser.parse_args()

    with open(args.svg, "r", encoding="utf-8") as handle:
        svg_text = handle.read()
    native = Image.open(
        io.BytesIO(cairosvg.svg2png(bytestring=svg_text.encode("utf-8")))
    )

    sizes = [size for size in args.sizes if size > 0]
    if not sizes:
        raise SystemExit("at least one positive size is required")

    width = len(sizes) * PANEL + (len(sizes) + 1) * PAD
    height = PANEL + CAPTION + 3 * PAD
    sheet = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(sheet)
    title_font = load_font(15)
    label_font = load_font(13)
    small_font = load_font(11)
    draw.text(
        (PAD, PAD // 2),
        f"same source diagram: {os.path.basename(args.svg)}",
        fill=FG,
        font=title_font,
    )

    for position, size in enumerate(sizes):
        resized = native.resize((size, size), Image.Resampling.LANCZOS)
        x = PAD + position * (PANEL + PAD)
        y = 2 * PAD + 18
        panel = fit(resized)
        sheet.paste(panel, (x, y))
        draw.rectangle([x, y, x + PANEL - 1, y + PANEL - 1], outline=BORDER)
        draw.text((x, y + PANEL + 8), f"IMAGE_SIZE = {size}", fill=FG, font=label_font)
        draw.text(
            (x, y + PANEL + 26),
            f"raster: {resized.size[0]} x {resized.size[1]}",
            fill=MUTED,
            font=small_font,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sheet.save(args.out)
    print(f"source   : {args.svg}")
    print(f"native   : {native.mode} {native.size}")
    print(f"sizes    : {sizes}")
    print(f"wrote    : {args.out}")


if __name__ == "__main__":
    main()
