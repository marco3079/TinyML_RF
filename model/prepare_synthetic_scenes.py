from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

IMAGE_SIZE = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera cenas de treino com zero a tres rostos em posicoes variadas."
    )
    parser.add_argument("--faces", type=Path, default=Path("model/dataset/rostos"))
    parser.add_argument("--output", type=Path, default=Path("model/dataset/cenas"))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_background(rng: np.random.Generator) -> Image.Image:
    base = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    start = rng.integers(20, 220, size=3)
    end = rng.integers(20, 220, size=3)
    for y in range(IMAGE_SIZE):
        ratio = y / (IMAGE_SIZE - 1)
        base[y, :, :] = start * (1.0 - ratio) + end * ratio
    noise = rng.normal(0, 10, base.shape)
    background = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(background, "RGBA")
    for _ in range(int(rng.integers(2, 8))):
        x1, y1 = rng.integers(-20, IMAGE_SIZE, size=2)
        width, height = rng.integers(10, 70, size=2)
        color = tuple(int(value) for value in rng.integers(0, 255, size=3)) + (int(rng.integers(20, 80)),)
        if rng.random() < 0.5:
            draw.rectangle((x1, y1, x1 + width, y1 + height), fill=color)
        else:
            draw.ellipse((x1, y1, x1 + width, y1 + height), fill=color)
    return background.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.0, 1.2))))


def overlaps(existing: list[tuple[int, int, int, int]], candidate: tuple[int, int, int, int]) -> bool:
    left, top, width, height = candidate
    for other_left, other_top, other_width, other_height in existing:
        intersection_width = max(0, min(left + width, other_left + other_width) - max(left, other_left))
        intersection_height = max(0, min(top + height, other_top + other_height) - max(top, other_top))
        intersection = intersection_width * intersection_height
        smaller_area = min(width * height, other_width * other_height)
        if smaller_area and intersection / smaller_area > 0.2:
            return True
    return False


def paste_face(
    canvas: Image.Image,
    source: Path,
    rng: np.random.Generator,
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    size = int(rng.integers(24, 58))
    for _ in range(30):
        left = int(rng.integers(0, IMAGE_SIZE - size + 1))
        top = int(rng.integers(0, IMAGE_SIZE - size + 1))
        candidate = (left, top, size, size)
        if not overlaps(boxes, candidate):
            break
    else:
        return False

    face = Image.open(source).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    if rng.random() < 0.5:
        face = face.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    face = ImageEnhance.Brightness(face).enhance(float(rng.uniform(0.75, 1.25)))
    face = ImageEnhance.Contrast(face).enhance(float(rng.uniform(0.8, 1.2)))
    if rng.random() < 0.35:
        face = face.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.2, 0.8))))

    mask = Image.new("L", (size, size), 0)
    mask_draw = ImageDraw.Draw(mask)
    inset = max(1, size // 24)
    mask_draw.rounded_rectangle((inset, inset, size - inset, size - inset), radius=size // 7, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1.0, size / 28)))
    canvas.paste(face, (left, top), mask)
    boxes.append(candidate)
    return True


def main() -> None:
    args = parse_args()
    sources = sorted(args.faces.glob("*.jpg"))
    if not sources:
        raise FileNotFoundError(f"Nenhum rosto encontrado em {args.faces}")
    args.output.mkdir(parents=True, exist_ok=True)
    for pattern in ("scene_*.jpg", "scene_*.txt"):
        for old_file in args.output.glob(pattern):
            old_file.unlink()

    rng = np.random.default_rng(args.seed)
    face_total = 0
    for index in range(args.count):
        canvas = make_background(rng)
        boxes: list[tuple[int, int, int, int]] = []
        face_count = int(rng.choice((0, 1, 2, 3), p=(0.22, 0.38, 0.28, 0.12)))
        selected = rng.choice(sources, size=face_count, replace=False)
        for source in selected:
            paste_face(canvas, Path(source), rng, boxes)

        image_path = args.output / f"scene_{index + 1:05d}.jpg"
        canvas.save(image_path, quality=88)
        labels = []
        for left, top, width, height in boxes:
            labels.append(
                f"0 {(left + width / 2) / IMAGE_SIZE:.7f} "
                f"{(top + height / 2) / IMAGE_SIZE:.7f} "
                f"{width / IMAGE_SIZE:.7f} {height / IMAGE_SIZE:.7f}"
            )
        image_path.with_suffix(".txt").write_text("\n".join(labels), encoding="ascii")
        face_total += len(boxes)
        if (index + 1) % 250 == 0:
            print(f"{index + 1}/{args.count} cenas")
    print(f"Geradas {args.count} cenas com {face_total} rostos em {args.output}")


if __name__ == "__main__":
    main()
