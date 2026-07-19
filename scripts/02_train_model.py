import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import Config
from utils.audio_dataset import AudioDataset
from models.model_trainer import ModelTrainer


def print_banner():
    print("\n" + "=" * 70)
    print("  DRONE SHIELD AI - YAMNet Fine-Tuning Pipeline")
    print("=" * 70)


def main():

    print_banner()

    Config.create_all_dirs()

    print("\n[STEP 1/6] Loading audio dataset...")
    print("-" * 70)

    dataset = AudioDataset(
        drone_dir=Config.DRONE_DIR,
        non_drone_dir=Config.NON_DRONE_DIR,
        sample_rate=Config.SAMPLE_RATE,
        duration=Config.DURATION
    )

    print("\n[STEP 2/6] Preparing and splitting data...")
    print("-" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test = dataset.prepare_dataset(
        test_size=Config.TEST_SIZE,
        val_size=Config.VAL_SIZE,
        random_state=Config.RANDOM_STATE
    )

    print("\n[STEP 3/6] Creating TensorFlow datasets...")
    print("-" * 70)

    train_tf = dataset.create_tf_dataset(
        X_train, y_train,
        batch_size=Config.BATCH_SIZE,
        shuffle=True
    )

    val_tf = dataset.create_tf_dataset(
        X_val, y_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    test_tf = dataset.create_tf_dataset(
        X_test, y_test,
        batch_size=Config.BATCH_SIZE,
        shuffle=False
    )

    print(f"  Train batches:      {len(train_tf)}")
    print(f"  Validation batches: {len(val_tf)}")
    print(f"  Test batches:       {len(test_tf)}")

    print("\n[STEP 4/6] Calculating class weights...")
    print("-" * 70)

    class_weights = dataset.get_class_weights(y_train)

    print("\n[STEP 5/6] Building and training model...")
    print("-" * 70)

    trainer = ModelTrainer(
        learning_rate=Config.LEARNING_RATE,
        model_dir=Config.MODELS_DIR
    )

    model = trainer.build_model()

    print("\n[STEP 6/6] Starting training...")
    print("-" * 70)

    history = trainer.train(
        train_dataset=train_tf,
        val_dataset=val_tf,
        epochs=Config.EPOCHS,
        class_weights=class_weights,
        verbose=1
    )

    trainer.save_training_history()

    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Best model saved: {Config.BEST_MODEL_PATH}")
    print(f"  Training logs:    {Config.LOGS_DIR}")
    print(f"\n  Next step: Run evaluation script")
    print(f"  python scripts/03_evaluate.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
