from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

from prepare_synthetic_scenes import IMAGE_SIZE, make_background


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera recortes reais em que 1-3 rostos permanecem visiveis na entrada 96x96."
    )
    parser.add_argument("--source", type=Path, default=Path("model/dataset/wider_real"))
    parser.add_argument("--output", type=Path, default=Path("model/dataset/real_crops"))
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--negative-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="ascii").splitlines():
        values = line.split()
        if len(values) == 5:
            boxes.append(tuple(map(float, values[1:])))
    return boxes


def transform_boxes(
    boxes: list[tuple[float, float, float, float]],
    crop_left: float,
    crop_top: float,
    crop_size: float,
) -> list[tuple[float, float, float, float]]:
    transformed = []
    occupied = set()
    for center_x, center_y, width, height in boxes:
        if not (crop_left <= center_x <= crop_left + crop_size and
                crop_top <= center_y <= crop_top + crop_size):
            continue
        new_box = (
            (center_x - crop_left) / crop_size,
            (center_y - crop_top) / crop_size,
            width / crop_size,
            height / crop_size,
        )
        if min(new_box[2], new_box[3]) * IMAGE_SIZE < 10 or max(new_box[2:]) > 0.9:
            continue
        cell = (int(new_box[0] * 10), int(new_box[1] * 10))
        if cell in occupied:
            continue
        occupied.add(cell)
        transformed.append(new_box)
    return transformed


def crop_around_faces(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    rng: np.random.Generator,
) -> tuple[Image.Image, list[tuple[float, float, float, float]]] | None:
    selected_count = min(len(boxes), int(rng.integers(1, 4)))
    selected_indices = rng.choice(len(boxes), size=selected_count, replace=False)
    selected = [boxes[int(index)] for index in selected_indices]
    left = min(center_x - width / 2 for center_x, _, width, _ in selected)
    top = min(center_y - height / 2 for _, center_y, _, height in selected)
    right = max(center_x + width / 2 for center_x, _, width, _ in selected)
    bottom = max(center_y + height / 2 for _, center_y, _, height in selected)
    crop_size = min(1.0, max(right - left, bottom - top) * float(rng.uniform(1.35, 2.0)))
    if crop_size <= 0.0:
        return None
    center_x = (left + right) / 2 + float(rng.uniform(-0.08, 0.08)) * crop_size
    center_y = (top + bottom) / 2 + float(rng.uniform(-0.08, 0.08)) * crop_size
    crop_left = min(max(0.0, center_x - crop_size / 2), 1.0 - crop_size)
    crop_top = min(max(0.0, center_y - crop_size / 2), 1.0 - crop_size)
    transformed = transform_boxes(boxes, crop_left, crop_top, crop_size)
    if not transformed or len(transformed) > 3:
        return None

    width, height = image.size
    pixels = (
        int(crop_left * width), int(crop_top * height),
        int((crop_left + crop_size) * width), int((crop_top + crop_size) * height),
    )
    crop = image.crop(pixels).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    if rng.random() < 0.5:
        crop = crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        transformed = [(1.0 - x, y, w, h) for x, y, w, h in transformed]
    crop = ImageEnhance.Brightness(crop).enhance(float(rng.uniform(0.8, 1.2)))
    return crop, transformed


def save_sample(output: Path, number: int, image: Image.Image,
                boxes: list[tuple[float, float, float, float]]) -> None:
    image_path = output / f"crop_{number:05d}.jpg"
    image.save(image_path, quality=90)
    labels = [f"0 {x:.7f} {y:.7f} {w:.7f} {h:.7f}" for x, y, w, h in boxes]
    image_path.with_suffix(".txt").write_text("\n".join(labels), encoding="ascii")


def generate_positive_samples(
    sources: list[Path], output: Path, start: int, count: int, rng: np.random.Generator
) -> int:
    number = start
    attempts = 0
    while number < count and attempts < count * 20:
        attempts += 1
        source = Path(rng.choice(sources))
        result = crop_around_faces(
            Image.open(source).convert("RGB"), read_boxes(source.with_suffix(".txt")), rng
        )
        if result is None:
            continue
        number += 1
        save_sample(output, number, *result)
        if number % 250 == 0:
            print(f"{number}/{count} recortes")
    return number


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for pattern in ("crop_*.jpg", "crop_*.txt"):
        for old_file in args.output.glob(pattern):
            old_file.unlink()
    rng = np.random.default_rng(args.seed)
    positive_sources = [
        path for path in args.source.glob("*.jpg")
        if path.with_suffix(".txt").read_text(encoding="ascii").strip()
    ]
    if not positive_sources:
        raise FileNotFoundError("Nenhuma cena WIDER positiva encontrada.")

    negative_count = int(args.count * args.negative_ratio)
    for number in range(1, negative_count + 1):
        save_sample(args.output, number, make_background(rng), [])
    generated = generate_positive_samples(
        positive_sources, args.output, negative_count, args.count, rng
    )
    if generated < args.count:
        raise RuntimeError(f"Foram gerados apenas {generated} de {args.count} recortes.")
    print(f"Concluido: {generated} recortes reais, incluindo {negative_count} negativos limpos")


if __name__ == "__main__":
    main()
