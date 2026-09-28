from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
from pathlib import Path

import numpy as np

IMAGE_SIZE = 96
GRID_SIZE = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina do zero um detector facial em grade e exporta pesos INT8."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("src/model_data.h"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def load_sample(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    from PIL import Image

    image = Image.open(image_path).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(image, dtype=np.float32)[..., None] / 255.0
    target = np.zeros((GRID_SIZE, GRID_SIZE, 5), dtype=np.float32)
    label_path = image_path.with_suffix(".txt")

    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            values = line.split()
            if len(values) != 5 or int(values[0]) != 0:
                continue
            center_x, center_y, width, height = map(float, values[1:])
            grid_x = min(GRID_SIZE - 1, max(0, int(center_x * GRID_SIZE)))
            grid_y = min(GRID_SIZE - 1, max(0, int(center_y * GRID_SIZE)))
            target[grid_y, grid_x] = (
                1.0,
                center_x * GRID_SIZE - grid_x,
                center_y * GRID_SIZE - grid_y,
                width,
                height,
            )
    return pixels, target


def load_dataset(root: Path) -> tuple[np.ndarray, np.ndarray]:
    image_paths = sorted(
        path
        for extension in ("*.jpg", "*.jpeg", "*.png")
        for path in root.rglob(extension)
    )
    if not image_paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {root}")
    samples = [load_sample(path) for path in image_paths]
    images, targets = zip(*samples, strict=True)
    return np.stack(images), np.stack(targets)


def build_model():
    import tensorflow as tf

    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 1), name="grayscale")
    value = tf.keras.layers.RandomContrast(0.2, name="augment_contrast")(inputs)
    value = tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0), name="augment_brightness")(value)
    value = tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu", name="conv1")(value)
    value = tf.keras.layers.Conv2D(16, 3, strides=2, activation="relu", name="conv2")(value)
    value = tf.keras.layers.Conv2D(24, 5, strides=2, activation="relu", name="conv3")(value)
    outputs = tf.keras.layers.Conv2D(5, 1, activation="sigmoid", name="head")(value)
    return tf.keras.Model(inputs, outputs, name="tiny_face_grid")


def detection_loss(y_true, y_pred):
    import tensorflow as tf

    object_true = y_true[..., 0]
    object_pred = y_pred[..., 0]
    object_loss = tf.keras.backend.binary_crossentropy(object_true, object_pred)
    object_weight = 1.0 + object_true * 49.0
    box_loss = tf.reduce_sum(tf.square(y_true[..., 1:] - y_pred[..., 1:]), axis=-1)
    return tf.reduce_mean(object_loss * object_weight + box_loss * object_true * 5.0, axis=None)


def quantize(values: np.ndarray) -> tuple[np.ndarray, float]:
    maximum = float(np.max(np.abs(values)))
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    return np.clip(np.rint(values / scale), -127, 127).astype(np.int8), scale


def c_array(name: str, values: np.ndarray) -> str:
    flattened = values.reshape(-1)
    rows = []
    for start in range(0, len(flattened), 16):
        rows.append("  " + ", ".join(str(int(value)) for value in flattened[start : start + 16]))
    return f"constexpr int8_t {name}[] = {{\n" + ",\n".join(rows) + "\n};\n"


def export_header(model, output: Path, metrics: dict[str, float]) -> None:
    blocks = [
        "#pragma once\n\n#include <stdint.h>\n\nnamespace face_model {\n",
        "constexpr bool trained = true;\n",
        f"constexpr int input_size = {IMAGE_SIZE};\n",
        f"constexpr int grid_size = {GRID_SIZE};\n",
        f"constexpr float validation_loss = {metrics['val_loss']:.8f}F;\n\n",
    ]
    for layer_name in ("conv1", "conv2", "conv3", "head"):
        kernel, bias = model.get_layer(layer_name).get_weights()
        quantized, scale = quantize(kernel)
        blocks.append(f"constexpr float {layer_name}_scale = {scale:.10g}F;\n")
        blocks.append(c_array(f"{layer_name}_weights", quantized))
        blocks.append(
            f"constexpr float {layer_name}_bias[] = {{"
            + ", ".join(f"{float(value):.10g}F" for value in bias)
            + "};\n\n"
        )
    blocks.append("}  // namespace face_model\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(blocks), encoding="ascii")


def main() -> None:
    args = parse_args()
    images, targets = load_dataset(args.dataset)
    face_count = int(np.sum(targets[..., 0]))
    print(f"Dataset: {len(images)} imagens, {face_count} faces rotuladas")
    if args.validate_only:
        return

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    indices = np.random.default_rng(args.seed).permutation(len(images))
    split = max(1, int(len(indices) * 0.8))
    train_indices, validation_indices = indices[:split], indices[split:]
    if len(validation_indices) == 0:
        raise ValueError("O dataset precisa ter pelo menos cinco imagens.")

    model = build_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=detection_loss)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
    ]
    history = model.fit(
        images[train_indices],
        targets[train_indices],
        validation_data=(images[validation_indices], targets[validation_indices]),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    metrics = {"val_loss": float(min(history.history["val_loss"]))}
    export_header(model, args.output, metrics)
    model.save(args.output.with_suffix(".keras"))
    print(f"Modelo exportado para {args.output}")


if __name__ == "__main__":
    main()