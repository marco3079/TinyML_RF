from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Converte WIDER FACE para rotulos YOLO locais.")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lines = args.annotations.read_text(encoding="utf-8").splitlines()
    index = 0
    image_count = 0
    face_count = 0

    while index < len(lines):
        relative_path = lines[index].strip()
        index += 1
        if not relative_path:
            continue
        box_count = int(lines[index])
        index += 1
        image_path = args.images / relative_path
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        labels = []
        for _ in range(box_count):
            values = [int(value) for value in lines[index].split()]
            index += 1
            x, y, width, height = values[:4]
            invalid = values[4] == 1
            if invalid or width < 8 or height < 8:
                continue
            center_x = (x + width / 2) / image_width
            center_y = (y + height / 2) / image_height
            labels.append(
                f"0 {center_x:.7f} {center_y:.7f} "
                f"{width / image_width:.7f} {height / image_height:.7f}"
            )

        image_path.with_suffix(".txt").write_text("\n".join(labels), encoding="ascii")
        image_count += 1
        face_count += len(labels)

    print(f"Convertidas {image_count} imagens e {face_count} faces")


if __name__ == "__main__":
    main()