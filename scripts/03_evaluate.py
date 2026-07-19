import sys
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import Config
from utils.audio_dataset import AudioDataset
from evaluation.metrics import MetricsCalculator
from tensorflow import keras

try:
    from models.model_trainer_unfreeze import YAMNetEmbedding
except ImportError:
    YAMNetEmbedding = None


def print_banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_model():
    model_path = Config.MODELS_DIR / 'best_yamnet_finetuned_unfreeze.keras'
    
    if not model_path.exists():
        model_path = Config.BEST_MODEL_PATH
    
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please run training first:")
        print("  python scripts/02_train_model.py")
        print("  or python scripts/07_retrain_unfreeze.py")
        sys.exit(1)

    print(f"\nLoading model from: {model_path}")
    
    custom_objects = {}
    if YAMNetEmbedding is not None:
        custom_objects = {'YAMNetEmbedding': YAMNetEmbedding}
    
    model = keras.models.load_model(str(model_path), custom_objects=custom_objects)
    print("Model loaded successfully")

    return model


def evaluate_on_dataset(model, X, y, dataset_name):
    print(f"\n{'─' * 70}")
    print(f"  Evaluating on: {dataset_name}")
    print(f"  Samples: {len(X)}")
    print(f"{'─' * 70}")

    y_pred_proba = model.predict(X, verbose=0).flatten()
    
    threshold = 0.3
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = MetricsCalculator.calculate_all_metrics(y, y_pred, y_pred_proba)
    MetricsCalculator.print_metrics(metrics)

    return metrics, y_pred, y_pred_proba


def save_results(metrics, dataset_name, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'dataset': dataset_name,
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'accuracy': float(metrics['accuracy']),
    }

    if 'mAP' in metrics:
        results['mAP'] = float(metrics['mAP'])

    results['confusion_matrix'] = {
        'true_negatives': int(metrics['true_negatives']),
        'false_positives': int(metrics['false_positives']),
        'false_negatives': int(metrics['false_negatives']),
        'true_positives': int(metrics['true_positives'])
    }

    output_file = output_dir / f'{dataset_name}_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {output_file}")


def main():

    print_banner("DRONE SHIELD AI - Model Evaluation")

    model = load_model()

    print("\n[STEP 1/3] Loading test data...")
    print("-" * 70)

    dataset = AudioDataset(
        drone_dir=Config.DRONE_DIR,
        non_drone_dir=Config.NON_DRONE_DIR,
        sample_rate=Config.SAMPLE_RATE,
        duration=Config.DURATION
    )

    X_train, y_train, X_val, y_val, X_test, y_test = dataset.prepare_dataset(
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE
    )

    print("\n[STEP 2/3] Internal Evaluation (same source)...")
    print("-" * 70)

    internal_metrics, _, _ = evaluate_on_dataset(
        model, X_test, y_test, "internal_test"
    )

    save_results(internal_metrics, "internal_test", Config.LOGS_DIR)

    print("\n[STEP 3/3] External Evaluation (different source)...")
    print("-" * 70)

    ext_drone_files = list(Config.EXTERNAL_DRONE_DIR.glob('*.wav'))
    ext_non_drone_files = list(Config.EXTERNAL_NON_DRONE_DIR.glob('*.wav'))
    total_external = len(ext_drone_files) + len(ext_non_drone_files)

    if total_external > 0:
        print(f"\nFound {total_external} external test files")
        print(f"  Drone: {len(ext_drone_files)}")
        print(f"  Non-Drone: {len(ext_non_drone_files)}")

        ext_dataset = AudioDataset(
            drone_dir=Config.EXTERNAL_DRONE_DIR,
            non_drone_dir=Config.EXTERNAL_NON_DRONE_DIR,
            sample_rate=Config.SAMPLE_RATE,
            duration=Config.DURATION
        )

        X_ext_drone = []
        for f in ext_drone_files:
            X_ext_drone.append(ext_dataset.load_audio_file(f))

        X_ext_non_drone = []
        for f in ext_non_drone_files:
            X_ext_non_drone.append(ext_dataset.load_audio_file(f))

        X_ext = np.array(X_ext_drone + X_ext_non_drone, dtype=np.float32)
        y_ext = np.array(
            [1] * len(X_ext_drone) + [0] * len(X_ext_non_drone),
            dtype=np.float32
        )

        external_metrics, _, _ = evaluate_on_dataset(
            model, X_ext, y_ext, "external_test"
        )

        save_results(external_metrics, "external_test", Config.LOGS_DIR)

    else:
        print("\nNo external test data found.")
        print(f"  Expected location: {Config.EXTERNAL_TEST_DIR}")
        print("\nTo add external test data:")
        print("  1. Download audio files from a different source")
        print("  2. Place drone sounds in: data/external_test/drone/")
        print("  3. Place other sounds in: data/external_test/non_drone/")
        print("  4. Run: python scripts/00_check_duplicates.py")
        print("  5. Re-run this evaluation script")

    print_banner("EVALUATION COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
