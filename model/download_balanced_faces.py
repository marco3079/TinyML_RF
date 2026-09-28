from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "HuggingFaceM4/FairFace"
CONFIG = "0.25"
SPLIT = "train"
GROUPS = (
    "East Asian",
    "Indian",
    "Black",
    "White",
    "Middle Eastern",
    "Latino_Hispanic",
    "Southeast Asian",
)
TARGETS = (15, 15, 14, 14, 14, 14, 14)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Baixa 100 rostos balanceados do FairFace para deteccao facial."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/dataset/rostos"),
    )
    return parser.parse_args()


def get_rows(offset: int, length: int = 100) -> list[dict]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": SPLIT,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        f"https://datasets-server.huggingface.co/rows?{query}",
        headers={"User-Agent": "TinyML-face-dataset/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["rows"]


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "TinyML-face-dataset/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        destination.write_bytes(response.read())


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for old_file in args.output.glob("fairface_*.jpg"):
        old_file.unlink()
        old_file.with_suffix(".txt").unlink(missing_ok=True)

    counts = [0] * len(GROUPS)
    records = []
    offset = 0

    while sum(counts) < sum(TARGETS):
        rows = get_rows(offset)
        if not rows:
            raise RuntimeError("O endpoint terminou antes de completar a amostra.")
        for item in rows:
            row = item["row"]
            group = int(row["race"])
            if counts[group] >= TARGETS[group]:
                continue
            image_number = sum(counts) + 1
            image_name = f"fairface_{image_number:03d}.jpg"
            image_path = args.output / image_name
            download(row["image"]["src"], image_path)
            image_path.with_suffix(".txt").write_text(
                "0 0.5 0.5 0.9 0.9\n", encoding="ascii"
            )
            counts[group] += 1
            records.append(
                {
                    "file": image_name,
                    "source_row": int(item["row_idx"]),
                    "source": DATASET,
                }
            )
            print(f"{sum(counts):3d}/100 {image_name}")
            if sum(counts) == sum(TARGETS):
                break
        offset += len(rows)

    (args.output / "manifest.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    distribution = ", ".join(
        f"{name}: {count}" for name, count in zip(GROUPS, counts, strict=True)
    )
    (args.output / "SOURCE.md").write_text(
        "# Fonte das imagens\n\n"
        "Amostra de 100 imagens do FairFace, distribuida entre os sete grupos "
        "declarados pelo dataset apenas para diversidade de coleta. O modelo usa "
        "somente a classe `face`; os rotulos demograficos nao sao exportados.\n\n"
        f"Distribuicao da amostra: {distribution}.\n\n"
        "Fonte: https://huggingface.co/datasets/HuggingFaceM4/FairFace\n\n"
        "Licenca: CC BY 4.0. Cite Karkkainen e Joo, FairFace, WACV 2021.\n",
        encoding="utf-8",
    )
    print(f"Concluido: {sum(counts)} imagens em {args.output}")


if __name__ == "__main__":
    main()
