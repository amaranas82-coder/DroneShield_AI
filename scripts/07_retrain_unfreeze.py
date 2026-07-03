import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tensorflow as tf
from utils.config import Config
from utils.audio_dataset import AudioDataset
from models.model_trainer_unfreeze import ModelTrainerUnfreeze

def main():
    Config.create_all_dirs()

    print("=" * 70)
    print("  RETRAINING WITH DEEP CLASSIFICATION HEAD")
    print("=" * 70)

    print("\n[STEP 1/6] Loading dataset...")
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
    train_ds = dataset.create_tf_dataset(
        X_train,
        y_train,
        batch_size=16,
        shuffle=True
    )

    val_ds = dataset.create_tf_dataset(
        X_val,
        y_val,
        batch_size=16,
        shuffle=False
    )

    print(f"  Train batches: {len(train_ds)}")
    print(f"  Val batches: {len(val_ds)}")

    print("\n[STEP 4/6] Calculating class weights...")
    print("-" * 70)
    class_weights = dataset.get_class_weights(y_train)

    print("\n[STEP 5/6] Building model...")
    print("-" * 70)
    trainer = ModelTrainerUnfreeze(
        learning_rate=1e-4, 
        batch_size=16,
        epochs=20,
        patience=6,
        model_dir=Config.MODELS_DIR
    )

    trainer.build_model()
    
    print("\n[STEP 6/6] Starting training...")
    print("-" * 70)
   
    trainer.train(train_dataset=train_ds, val_dataset=val_ds, class_weights=class_weights)
    trainer.save_training_history()

    print("\n" + "=" * 70)
    print("  RETRAINING PIPELINE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Evaluate the new model:")
    print("     python3 scripts/03_evaluate.py")
    print("\n  2. Compare with old model results")
    print("=" * 70)

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
