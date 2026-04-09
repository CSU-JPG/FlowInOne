#!/usr/bin/env python3
"""
Generate visual prompt input images for FlowInOne inference.

Takes source images and overlays instruction text, red markers, bounding boxes,
or creates text-on-color backgrounds, matching the format expected by the model.

Usage:
    python tests/prepare_visual_prompts.py --config tests/test_configs.json
    python tests/prepare_visual_prompts.py --config tests/test_configs.json --test-ids carnival_01_steampunk,kirby_04_t2i
"""

import argparse
import json
import math
import os
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# Font search paths (bold serif preferred, matching demo images)
FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "/System/Library/Fonts/Times.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/timesbd.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]


def find_font(size: int) -> ImageFont.FreeTypeFont:
    """Find an available font with fallback to PIL default."""
    for path in FONT_SEARCH_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
    print("  Warning: No TTF fonts found, using PIL default font. Text may be small.")
    return ImageFont.load_default()


def load_and_resize(image_path: str, target_size: int = 256) -> Image.Image:
    """Load image and center-crop to target_size, matching FlowInOne's preprocessing."""
    img = Image.open(image_path).convert("RGB")

    # Downscale progressively (matching center_crop_arr in i2i.py)
    while min(img.size) >= 2 * target_size:
        img = img.resize(
            tuple(x // 2 for x in img.size), resample=Image.BOX
        )
    scale = target_size / min(img.size)
    img = img.resize(
        tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC
    )

    # Center crop
    w, h = img.size
    left = (w - target_size) // 2
    top = (h - target_size) // 2
    img = img.crop((left, top, left + target_size, top + target_size))
    return img


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """Word-wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = " ".join(current_line + [word])
        try:
            bbox = font.getbbox(test_line)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            text_width = font.getsize(test_line)[0]

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def draw_text_with_outline(draw: ImageDraw.Draw, position: tuple, text: str,
                           font: ImageFont.FreeTypeFont,
                           fill: str = "white", outline: str = "black",
                           outline_width: int = 1):
    """Draw text with a thin outline for contrast."""
    x, y = position
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def draw_text_overlay(image: Image.Image, text: str, font_size: int = 15,
                      position: str = "bottom") -> Image.Image:
    """Draw instruction text on the bottom of an image with outline for readability."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    font = find_font(font_size)
    w, h = img.size
    padding = 6
    max_text_width = w - 2 * padding

    lines = wrap_text(text, font, max_text_width)

    # Calculate total text height
    try:
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 2
    except AttributeError:
        line_height = font.getsize("Ag")[1] + 2

    total_text_height = len(lines) * line_height + 2 * padding

    if position == "bottom":
        text_y_start = h - total_text_height
    elif position == "center":
        text_y_start = (h - total_text_height) // 2
    else:
        text_y_start = padding

    # Draw semi-transparent background bar
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [0, text_y_start - padding, w, h if position == "bottom" else text_y_start + total_text_height + padding],
        fill=(0, 0, 0, 100)
    )
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw text lines
    y = text_y_start
    for line in lines:
        try:
            bbox = font.getbbox(line)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            text_width = font.getsize(line)[0]

        x = (w - text_width) // 2
        draw_text_with_outline(draw, (x, y), line, font, fill="white", outline="black", outline_width=1)
        y += line_height

    return img


def create_text_in_image(source_path: str, instruction: str, output_path: str,
                         resolution: int = 256) -> str:
    """Create a visual prompt with text overlaid on the source image."""
    img = load_and_resize(source_path, resolution)
    img = draw_text_overlay(img, instruction, font_size=15, position="bottom")
    img.save(output_path)
    return output_path


def create_text2image(instruction: str, output_path: str, resolution: int = 256,
                      bg_color: tuple = (255, 105, 180)) -> str:
    """Create a text-on-colored-background image for text-to-image generation."""
    img = Image.new("RGB", (resolution, resolution), bg_color)
    draw = ImageDraw.Draw(img)
    font_size = 24
    font = find_font(font_size)
    padding = 15
    max_width = resolution - 2 * padding

    lines = wrap_text(instruction, font, max_width)

    try:
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + 4
    except AttributeError:
        line_height = font.getsize("Ag")[1] + 4

    total_height = len(lines) * line_height
    y_start = (resolution - total_height) // 2

    for i, line in enumerate(lines):
        try:
            bbox = font.getbbox(line)
            text_width = bbox[2] - bbox[0]
        except AttributeError:
            text_width = font.getsize(line)[0]

        x = (resolution - text_width) // 2
        draw.text((x, y_start + i * line_height), line, fill="black", font=font)

    img.save(output_path)
    return output_path


def create_visual_marker(source_path: str, instruction: str, output_path: str,
                         arrow_positions: list = None, resolution: int = 256) -> str:
    """Create a visual prompt with red arrows and text overlay."""
    img = load_and_resize(source_path, resolution)
    draw = ImageDraw.Draw(img)

    # Default arrow: point to center of image
    if not arrow_positions:
        arrow_positions = [{"start": [0.3, 0.15], "end": [0.5, 0.35]}]

    # Draw red arrows
    for arrow in arrow_positions:
        sx = int(arrow["start"][0] * resolution)
        sy = int(arrow["start"][1] * resolution)
        ex = int(arrow["end"][0] * resolution)
        ey = int(arrow["end"][1] * resolution)

        # Draw arrow line
        draw.line([(sx, sy), (ex, ey)], fill="red", width=3)

        # Draw arrowhead
        angle = math.atan2(ey - sy, ex - sx)
        head_length = 10
        for offset_angle in [2.5, -2.5]:
            hx = ex - head_length * math.cos(angle + offset_angle * 0.3)
            hy = ey - head_length * math.sin(angle + offset_angle * 0.3)
            draw.line([(ex, ey), (int(hx), int(hy))], fill="red", width=3)

    img = draw_text_overlay(img, instruction, font_size=14, position="bottom")
    img.save(output_path)
    return output_path


def create_text_box_edit(source_path: str, label: str, bbox: list, output_path: str,
                         resolution: int = 256) -> str:
    """Create a visual prompt with a red bounding box and label."""
    img = load_and_resize(source_path, resolution)
    draw = ImageDraw.Draw(img)

    # Convert relative bbox [x1, y1, x2, y2] to absolute pixels
    x1 = int(bbox[0] * resolution)
    y1 = int(bbox[1] * resolution)
    x2 = int(bbox[2] * resolution)
    y2 = int(bbox[3] * resolution)

    # Draw red bounding box
    for offset in range(3):
        draw.rectangle(
            [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
            outline="red"
        )

    # Draw label inside box
    font = find_font(12)
    label_x = x1 + 4
    label_y = y1 + 4
    draw_text_with_outline(
        draw, (label_x, label_y), label, font,
        fill="red", outline="white", outline_width=1
    )

    img.save(output_path)
    return output_path


def create_doodles(source_path: str, instruction: str, output_path: str,
                   resolution: int = 256) -> str:
    """Create a doodle-style visual prompt (same as text_in_image but for doodle tasks)."""
    return create_text_in_image(source_path, instruction, output_path, resolution)


def process_test_case(case: dict, source_images: dict, repo_root: str,
                      output_dir: str, resolution: int = 256) -> str:
    """Process a single test case and return the output path."""
    task_type = case["task_type"]
    test_id = case["id"]
    instruction = case["instruction"]

    # Determine output subdirectory based on task type
    if case.get("skip_cross_atten", False) or task_type == "text2image":
        sub_dir = os.path.join(output_dir, "text2image")
    else:
        sub_dir = os.path.join(output_dir, "editing")

    os.makedirs(sub_dir, exist_ok=True)
    output_path = os.path.join(sub_dir, f"{test_id}.png")

    # Resolve source image path
    source_key = case.get("source_image")
    source_path = None
    if source_key and source_key in source_images:
        source_path = os.path.join(repo_root, source_images[source_key])
        if not os.path.exists(source_path):
            print(f"  WARNING: Source image not found: {source_path}")
            return None

    if task_type == "text_in_image":
        create_text_in_image(source_path, instruction, output_path, resolution)
    elif task_type == "text2image":
        bg_color = tuple(case.get("bg_color", [255, 105, 180]))
        create_text2image(instruction, output_path, resolution, bg_color)
    elif task_type == "visual_marker":
        arrows = case.get("arrow_positions", None)
        create_visual_marker(source_path, instruction, output_path, arrows, resolution)
    elif task_type == "text_box_edit":
        bbox = case.get("bbox", [0.6, 0.1, 0.9, 0.4])
        create_text_box_edit(source_path, instruction, bbox, output_path, resolution)
    elif task_type == "doodles":
        create_doodles(source_path, instruction, output_path, resolution)
    else:
        print(f"  WARNING: Unknown task type '{task_type}' for test {test_id}")
        return None

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate visual prompt input images for FlowInOne inference."
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to test_configs.json"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for prepared inputs (default: tests/prepared_inputs)"
    )
    parser.add_argument(
        "--resolution", type=int, default=256,
        help="Output image resolution (default: 256)"
    )
    parser.add_argument(
        "--test-ids", default=None,
        help="Comma-separated list of test IDs to process (default: all)"
    )
    args = parser.parse_args()

    # Determine repo root (parent of tests/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(args.config)))
    if not os.path.exists(os.path.join(repo_root, "i2i.py")):
        repo_root = os.path.dirname(os.path.abspath(args.config))
    if not os.path.exists(os.path.join(repo_root, "i2i.py")):
        repo_root = os.getcwd()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    source_images = config["source_images"]
    test_cases = config["test_cases"]
    resolution = args.resolution or config["global_settings"].get("resolution", 256)

    output_dir = args.output_dir or os.path.join(repo_root, "tests", "prepared_inputs")
    os.makedirs(output_dir, exist_ok=True)

    # Filter test cases if --test-ids specified
    if args.test_ids:
        filter_ids = set(args.test_ids.split(","))
        test_cases = [tc for tc in test_cases if tc["id"] in filter_ids]
        if not test_cases:
            print(f"ERROR: No test cases match IDs: {args.test_ids}")
            sys.exit(1)

    print(f"Preparing {len(test_cases)} visual prompt images...")
    print(f"  Resolution: {resolution}x{resolution}")
    print(f"  Output dir: {output_dir}")
    print()

    success = 0
    failed = 0
    manifest = []

    for case in test_cases:
        test_id = case["id"]
        print(f"  [{test_id}] {case['task_type']}: {case['instruction'][:60]}...")

        try:
            output_path = process_test_case(case, source_images, repo_root, output_dir, resolution)
            if output_path:
                manifest.append({
                    "id": test_id,
                    "task_type": case["task_type"],
                    "input_path": output_path,
                    "skip_cross_atten": case.get("skip_cross_atten", False),
                    "cfg_scale": case.get("cfg_scale", 7.0),
                    "sample_steps": case.get("sample_steps", 50),
                })
                print(f"    -> {output_path}")
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print()
    print(f"Done! {success} succeeded, {failed} failed.")
    print(f"Manifest: {manifest_path}")

    editing_dir = os.path.join(output_dir, "editing")
    t2i_dir = os.path.join(output_dir, "text2image")
    editing_count = len(os.listdir(editing_dir)) if os.path.exists(editing_dir) else 0
    t2i_count = len(os.listdir(t2i_dir)) if os.path.exists(t2i_dir) else 0
    print(f"  Editing inputs: {editing_count} images in {editing_dir}")
    print(f"  Text2Image inputs: {t2i_count} images in {t2i_dir}")


if __name__ == "__main__":
    main()
