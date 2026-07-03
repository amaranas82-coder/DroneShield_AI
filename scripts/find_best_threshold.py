import sys
from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.audio_dataset import AudioDataset


def load_trained_model(model_path: Path) -> keras.Model:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    class YAMNetEmbedding(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.yamnet = yamnet_model

        def call(self, inputs):
            def process_single_audio(audio):
                _, embeddings, _ = self.yamnet(audio)
                return tf.reduce_mean(embeddings, axis=0)

            embeddings_batch = tf.map_fn(
                process_single_audio,
                inputs,
                dtype=tf.float32
            )
            return embeddings_batch

    custom_objects = {"YAMNetEmbedding": YAMNetEmbedding}

    model = keras.models.load_model(
        str(model_path),
        custom_objects=custom_objects,
        compile=False
    )
    return model


def compute_metrics_from_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }


def main():
    random.seed(42)
    np.random.seed(42)

    model_path = Config.BEST_MODEL_PATH
    model = load_trained_model(model_path)

    n_per_class = 1000
    threshold_grid = np.arange(0.05, 0.95 + 1e-9, 0.01)

    ext_drone_dir = Config.EXTERNAL_DRONE_DIR
    ext_non_drone_dir = Config.EXTERNAL_NON_DRONE_DIR

    if not ext_drone_dir.exists() or not ext_non_drone_dir.exists():
        raise FileNotFoundError("External test directories are missing. Check Config paths.")

    dataset = AudioDataset(
        drone_dir=ext_drone_dir,
        non_drone_dir=ext_non_drone_dir,
        sample_rate=Config.SAMPLE_RATE,
        duration=Config.DURATION
    )

    drone_files = sorted(list(ext_drone_dir.glob("*.wav")))
    non_drone_files = sorted(list(ext_non_drone_dir.glob("*.wav")))

    if len(drone_files) == 0 or len(non_drone_files) == 0:
        raise RuntimeError("No wav files found in external_test directories.")

    random.shuffle(drone_files)
    random.shuffle(non_drone_files)

    drone_files = drone_files[:min(n_per_class, len(drone_files))]
    non_drone_files = non_drone_files[:min(n_per_class, len(non_drone_files))]

    if len(drone_files) < n_per_class or len(non_drone_files) < n_per_class:
        print("Warning: external_test has less than 1000 samples for one class.")
        print(f"Drone selected: {len(drone_files)}")
        print(f"Non-drone selected: {len(non_drone_files)}")

    X = []
    y_true = []

    for f in drone_files:
        X.append(dataset.load_audio_file(f))
        y_true.append(1)

    for f in non_drone_files:
        X.append(dataset.load_audio_file(f))
        y_true.append(0)

    X = np.asarray(X, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int32)

    batch_size = 32
    y_prob = model.predict(X, batch_size=batch_size, verbose=1).reshape(-1)

    results = []
    best_by_f1 = None

    for thr in threshold_grid:
        metrics = compute_metrics_from_threshold(y_true, y_prob, float(thr))
        results.append(metrics)

        if best_by_f1 is None or metrics["f1_score"] > best_by_f1["f1_score"]:
            best_by_f1 = metrics

    output = {
        "model_path": str(model_path),
        "external_test_selected": {
            "drone_count": int(np.sum(y_true == 1)),
            "non_drone_count": int(np.sum(y_true == 0))
        },
        "best_threshold": best_by_f1,
        "all_results": results
    }

    out_file = Config.RESULTS_DIR / "logs" / "threshold_optimization_external_1000.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print("Threshold optimization complete.")
    print(f"Best threshold: {best_by_f1['threshold']}")
    print(f"Precision: {best_by_f1['precision']}")
    print(f"Recall: {best_by_f1['recall']}")
    print(f"F1-Score: {best_by_f1['f1_score']}")
    print(f"Accuracy: {best_by_f1['accuracy']}")
    print(f"Confusion matrix: TP={best_by_f1['tp']} TN={best_by_f1['tn']} FP={best_by_f1['fp']} FN={best_by_f1['fn']}")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
