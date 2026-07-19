import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from pathlib import Path
import json
from datetime import datetime


class ModelTrainer:
    
    def __init__(self, learning_rate=1e-4, model_dir=None):
        self.learning_rate = learning_rate
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.history = None
        
        print("ModelTrainer initialized")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Model directory: {self.model_dir}")
    
    def build_model(self):
        print("\nBuilding model architecture...")
        print("-" * 60)

        print("Loading YAMNet from TensorFlow Hub...")
        
        yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        class YAMNetEmbedding(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.yamnet = yamnet_model

            def call(self, inputs):
                def process_single_audio(audio):
                    scores, embeddings, spectrogram = self.yamnet(audio)
                    return tf.reduce_mean(embeddings, axis=0)

                embeddings = tf.map_fn(
                    process_single_audio,
                    inputs,
                    dtype=tf.float32
                )
                return embeddings

        inputs = keras.Input(shape=(16000,), dtype=tf.float32, name='audio_input')

        embeddings = YAMNetEmbedding()(inputs)

        x = keras.layers.Dense(256, activation='relu', name='dense_1')(embeddings)
        x = keras.layers.Dropout(0.5, name='dropout_1')(x)

        x = keras.layers.Dense(128, activation='relu', name='dense_2')(x)
        x = keras.layers.Dropout(0.3, name='dropout_2')(x)

        outputs = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='yamnet_finetuned')

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

        print("\nModel compiled successfully!")
        print(f"Total parameters: {self.model.count_params():,}")

        return self.model
    
    def get_callbacks(self, checkpoint_dir=None, patience=10):
        if checkpoint_dir is None:
            checkpoint_dir = self.model_dir / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = []
        
        best_model_path = self.model_dir / 'best_yamnet_finetuned.h5'
        checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        log_path = self.model_dir.parent / 'results' / 'logs' / 'training.csv'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger = keras.callbacks.CSVLogger(str(log_path), append=False)
        callbacks.append(csv_logger)
        
        print("\nCallbacks configured:")
        print(f"  - ModelCheckpoint: {best_model_path}")
        print(f"  - EarlyStopping: patience={patience}")
        print(f"  - ReduceLROnPlateau: factor=0.5, patience=5")
        print(f"  - CSVLogger: {log_path}")
        
        return callbacks
    
    def train(self, train_dataset, val_dataset, epochs=50, 
              class_weights=None, verbose=1):
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print(f"Class weights: {class_weights}")
        print("-" * 60)
        
        callbacks = self.get_callbacks()
        
        start_time = datetime.now()
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=verbose
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Duration: {duration}")
        print(f"Final training loss: {self.history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {self.history.history['val_loss'][-1]:.4f}")
        print(f"Best validation loss: {min(self.history.history['val_loss']):.4f}")
        
        return self.history
    
    def save_training_history(self, output_path=None):
        if self.history is None:
            print("No training history to save.")
            return
        
        if output_path is None:
            output_path = self.model_dir.parent / 'results' / 'logs' / 'training_history.json'
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_dict = {
            'loss': [float(x) for x in self.history.history['loss']],
            'val_loss': [float(x) for x in self.history.history['val_loss']],
            'accuracy': [float(x) for x in self.history.history['accuracy']],
            'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
            'precision': [float(x) for x in self.history.history['precision']],
            'val_precision': [float(x) for x in self.history.history['val_precision']],
            'recall': [float(x) for x in self.history.history['recall']],
            'val_recall': [float(x) for x in self.history.history['val_recall']]
        }
        
        with open(output_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"\nTraining history saved: {output_path}")
    
    def print_model_summary(self):
        if self.model:
            self.model.summary()
        else:
            print("Model not built yet.")


if __name__ == "__main__":
    print("ModelTrainer Module - Test Mode")
    print("=" * 60)
    
    trainer = ModelTrainer(learning_rate=1e-4)
    model = trainer.build_model()
    
    print("\nModel Summary:")
    trainer.print_model_summary()
    
    print("\nModelTrainer ready for use!")
