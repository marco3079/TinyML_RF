from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

DATASET = "zhouxzh/retinaface_widerface"
CONFIG = "default"
IMAGE_SIZE = 96
GRID_SIZE = 10
QUOTAS = {0: 500, 1: 1000, 2: 700, 3: 300}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa uma amostra WIDER FACE real, balanceada por quantidade de rostos."
    )
    parser.add_argument("--output", type=Path, default=Path("model/dataset/wider_real"))
    parser.add_argument("--count", type=int, default=sum(QUOTAS.values()))
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    return parser.parse_args()


def get_rows(offset: int, split: str, length: int = 100) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={"User-Agent": "TinyML-WIDER-subset/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["rows"]


def download_image(url: str) -> Image.Image:
    request = urllib.request.Request(url, headers={"User-Agent": "TinyML-WIDER-subset/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return Image.open(io.BytesIO(response.read())).convert("RGB")


def usable_boxes(row: dict) -> list[tuple[float, float, float, float]] | None:
    image_width = float(row["width"])
    image_height = float(row["height"])
    boxes = []
    occupied_cells = set()
    values = row["bboxes"]
    for x, y, width, height in zip(
        values["x"], values["y"], values["w"], values["h"], strict=True
    ):
        x = max(0.0, float(x))
        y = max(0.0, float(y))
        width = min(float(width), image_width - x)
        height = min(float(height), image_height - y)
        normalized_width = width / image_width
        normalized_height = height / image_height
        if min(normalized_width, normalized_height) * IMAGE_SIZE < 8.0:
            continue
        center_x = (x + width / 2) / image_width
        center_y = (y + height / 2) / image_height
        cell = (
            min(GRID_SIZE - 1, int(center_x * GRID_SIZE)),
            min(GRID_SIZE - 1, int(center_y * GRID_SIZE)),
        )
        if cell in occupied_cells:
            return None
        occupied_cells.add(cell)
        boxes.append((center_x, center_y, normalized_width, normalized_height))
    return boxes if len(boxes) <= 3 else None


def scaled_quotas(count: int) -> dict[int, int]:
    if count == sum(QUOTAS.values()):
        return QUOTAS.copy()
    weights = {key: value / sum(QUOTAS.values()) for key, value in QUOTAS.items()}
    quotas = {key: int(count * weight) for key, weight in weights.items()}
    quotas[1] += count - sum(quotas.values())
    return quotas


def save_row(item: dict, boxes: list[tuple[float, float, float, float]],
             output: Path, number: int) -> dict | None:
    try:
        image = download_image(item["row"]["image"]["src"])
    except Exception as error:
        print(f"Ignorando linha {item['row_idx']}: {error}")
        return None
    image_name = f"wider_{number:05d}.jpg"
    image_path = output / image_name
    image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS).save(
        image_path, quality=90
    )
    labels = [
        f"0 {center_x:.7f} {center_y:.7f} {width:.7f} {height:.7f}"
        for center_x, center_y, width, height in boxes
    ]
    image_path.with_suffix(".txt").write_text("\n".join(labels), encoding="ascii")
    return {"file": image_name, "source_row": int(item["row_idx"])}


def collect_page(rows: list[dict], output: Path, count: int, quotas: dict[int, int],
                 counts: dict[int, int], records: list[dict]) -> None:
    for item in rows:
        boxes = usable_boxes(item["row"])
        category = len(boxes) if boxes is not None else -1
        if category not in quotas or counts[category] >= quotas[category]:
            continue
        number = sum(counts.values()) + 1
        record = save_row(item, boxes, output, number)
        if record is None:
            continue
        counts[category] += 1
        records.append(record)
        if number % 100 == 0:
            print(f"{number}/{count} imagens | {counts}")
        if number == count:
            return


def collect_rows(output: Path, count: int, split: str,
                 quotas: dict[int, int]) -> tuple[dict[int, int], list[dict]]:
    counts = dict.fromkeys(quotas, 0)
    records = []
    offset = 0
    while sum(counts.values()) < count and offset < 12880:
        rows = get_rows(offset, split)
        if not rows:
            break
        collect_page(rows, output, count, quotas, counts, records)
        offset += len(rows)
    if sum(counts.values()) < count:
        raise RuntimeError(f"Fim do split antes de completar as cotas: {counts}")
    return counts, records


def main() -> None:
    args = parse_args()
    quotas = scaled_quotas(args.count)
    args.output.mkdir(parents=True, exist_ok=True)
    for pattern in ("wider_*.jpg", "wider_*.txt"):
        for old_file in args.output.glob(pattern):
            old_file.unlink()

    counts, records = collect_rows(args.output, args.count, args.split, quotas)

    (args.output / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "SOURCE.md").write_text(
        "# Fonte\n\n"
        "Amostra derivada do WIDER FACE, usando cenas e bounding boxes reais.\n\n"
        "Espelho: https://huggingface.co/datasets/zhouxzh/retinaface_widerface\n\n"
        "Dataset original: http://shuoyang1213.me/WIDERFACE/\n\n"
        "Licenca: CC BY-NC-ND 4.0. Uso educacional e nao comercial.\n\n"
        f"Distribuicao: {counts}. Rostos menores que 8 px na entrada 96x96 foram ignorados.\n",
        encoding="utf-8",
    )
    print(f"Concluido: {sum(counts.values())} imagens em {args.output}")


if __name__ == "__main__":
    main()