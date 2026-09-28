from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false

import argparse
from pathlib import Path

import numpy as np

from train import GRID_SIZE, IMAGE_SIZE, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia localizacao e contagem facial.")
    parser.add_argument("--model", type=Path, default=Path("src/model_data.keras"))
    parser.add_argument("--dataset", type=Path, required=True)
    return parser.parse_args()


def iou(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    intersection_left = max(left[0], right[0])
    intersection_top = max(left[1], right[1])
    intersection_right = min(left[0] + left[2], right[0] + right[2])
    intersection_bottom = min(left[1] + left[3], right[1] + right[3])
    intersection = max(0.0, intersection_right - intersection_left) * max(
        0.0, intersection_bottom - intersection_top
    )
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def decode_prediction(prediction: np.ndarray, threshold: float) -> list[tuple[float, ...]]:
    candidates = []
    for grid_y in range(GRID_SIZE):
        for grid_x in range(GRID_SIZE):
            score, offset_x, offset_y, width, height = prediction[grid_y, grid_x]
            if score < threshold:
                continue
            top = max(0, grid_y - 1)
            bottom = min(GRID_SIZE, grid_y + 2)
            left = max(0, grid_x - 1)
            right = min(GRID_SIZE, grid_x + 2)
            if score < np.max(prediction[top:bottom, left:right, 0]):
                continue
            center_x = (grid_x + offset_x) / GRID_SIZE
            center_y = (grid_y + offset_y) / GRID_SIZE
            candidates.append(
                (center_x - width / 2, center_y - height / 2, width, height, float(score))
            )
    candidates.sort(key=lambda box: box[4], reverse=True)
    selected = []
    for candidate in candidates:
        if all(iou(candidate, previous) < 0.3 for previous in selected):
            selected.append(candidate)
    return selected


def decode_target(target: np.ndarray) -> list[tuple[float, ...]]:
    boxes = []
    for grid_y in range(GRID_SIZE):
        for grid_x in range(GRID_SIZE):
            present, offset_x, offset_y, width, height = target[grid_y, grid_x]
            if present < 0.5:
                continue
            center_x = (grid_x + offset_x) / GRID_SIZE
            center_y = (grid_y + offset_y) / GRID_SIZE
            boxes.append((center_x - width / 2, center_y - height / 2, width, height))
    return boxes


def evaluate(predictions: np.ndarray, targets: np.ndarray, threshold: float) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    exact_count = empty_scenes = empty_scene_errors = 0
    matched_ious = []
    for prediction, target in zip(predictions, targets, strict=True):
        predicted_boxes = decode_prediction(prediction, threshold)
        target_boxes = decode_target(target)
        exact_count += len(predicted_boxes) == len(target_boxes)
        if not target_boxes:
            empty_scenes += 1
            empty_scene_errors += bool(predicted_boxes)

        unmatched = set(range(len(target_boxes)))
        for predicted_box in predicted_boxes:
            matches = [(iou(predicted_box, target_boxes[index]), index) for index in unmatched]
            best_iou, best_index = max(matches, default=(0.0, -1))
            if best_iou >= 0.3:
                true_positive += 1
                matched_ious.append(best_iou)
                unmatched.remove(best_index)
            else:
                false_positive += 1
        false_negative += len(unmatched)

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(1e-9, precision + recall),
        "count_accuracy": exact_count / len(targets),
        "empty_false_positive": empty_scene_errors / max(1, empty_scenes),
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }


def main() -> None:
    args = parse_args()
    import tensorflow as tf

    images, targets = load_dataset(args.dataset)
    model = tf.keras.models.load_model(args.model, compile=False)
    predictions = model.predict(images, batch_size=64, verbose=0)
    print(f"Avaliacao: {len(images)} cenas, entrada {IMAGE_SIZE}x{IMAGE_SIZE}")
    print("threshold precision recall f1 count_acc empty_fp mean_iou")
    results = []
    for threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        result = evaluate(predictions, targets, threshold)
        results.append(result)
        print(
            f"{threshold:.2f} {result['precision']:.3f} {result['recall']:.3f} "
            f"{result['f1']:.3f} {result['count_accuracy']:.3f} "
            f"{result['empty_false_positive']:.3f} {result['mean_iou']:.3f}"
        )
    best = max(results, key=lambda item: item["f1"])
    print(f"Melhor threshold por F1: {best['threshold']:.2f}")


if __name__ == "__main__":
    main()