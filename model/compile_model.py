from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
import re
from pathlib import Path

from train import detection_loss, export_header


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantiza e exporta um modelo Keras treinado.")
    parser.add_argument("--model", type=Path, default=Path("src/model_data.keras"))
    parser.add_argument("--output", type=Path, default=Path("src/model_data.h"))
    return parser.parse_args()


def previous_validation_loss(header: Path) -> float:
    if not header.exists():
        return 0.0
    match = re.search(r"validation_loss\s*=\s*([0-9.eE+-]+)F", header.read_text())
    return float(match.group(1)) if match else 0.0


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    if not args.model.exists():
        raise FileNotFoundError(f"Modelo nao encontrado: {args.model}")
    model = tf.keras.models.load_model(
        args.model,
        custom_objects={"detection_loss": detection_loss},
        compile=False,
    )
    export_header(
        model,
        args.output,
        {"val_loss": previous_validation_loss(args.output)},
    )
    print(f"Modelo compilado para {args.output}")


if __name__ == "__main__":
    main()
