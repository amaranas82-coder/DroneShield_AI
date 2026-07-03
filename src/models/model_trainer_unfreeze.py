import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from pathlib import Path
import json
from datetime import datetime

# 1. تسجيل الطبقة برمجياً لحل مشكلة الحفظ والـ Serialization
@keras.utils.register_keras_serializable(package="CustomLayers")
class YAMNetEmbedding(keras.layers.Layer):
    def __init__(self, hub_url='https://tfhub.dev/google/yamnet/1', **kwargs):
        super().__init__(**kwargs)
        self.hub_url = hub_url
        # تحميل النموذج داخلياً بناءً على الرابط النصي
        self.yamnet_hub = hub.load(self.hub_url)

    def call(self, inputs):
        def process_single_audio(audio):
            scores, embeddings, spectrogram = self.yamnet_hub(audio)
            return tf.reduce_mean(embeddings, axis=0)

        embeddings = tf.map_fn(
            process_single_audio,
            inputs,
            dtype=tf.float32
        )
        return embeddings

    # 2. إتاحة دالة الـ config التي طلبها كيرس في رسالة الخطأ
    def get_config(self):
        config = super().get_config()
        config.update({
            "hub_url": self.hub_url
        })
        return config


class ModelTrainerUnfreeze:
    
    def __init__(self, learning_rate=1e-4, batch_size=16, epochs=20, patience=6, model_dir=None):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.model_dir = model_dir or Path("models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.history = None
        
        print("ModelTrainerUnfreeze initialized")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Batch size: {self.batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Patience: {self.patience}")
        print(f"Model directory: {self.model_dir}")
    
    def build_model(self):
        print("\nBuilding model with Deep Classification Head...")
        print("-" * 60)

        inputs = keras.Input(shape=(16000,), dtype=tf.float32, name='audio_input')

        # تمرير الـ URL كنص قابل للتسلسل والحفظ برمجياً
        embeddings = YAMNetEmbedding(hub_url='https://tfhub.dev/google/yamnet/1')(inputs)

        # --- رأس تصنيف عميق لرفع دقة الـ F1-Score وعزل الضوضاء ---
        x = keras.layers.Dense(512, name='dense_1')(embeddings)
        x = keras.layers.BatchNormalization(name='batch_norm_1')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.5, name='dropout_1')(x)

        x = keras.layers.Dense(256, name='dense_2')(x)
        x = keras.layers.BatchNormalization(name='batch_norm_2')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.4, name='dropout_2')(x)

        x = keras.layers.Dense(64, name='dense_3')(x)
        x = keras.layers.BatchNormalization(name='batch_norm_3')(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Dropout(0.2, name='dropout_3')(x)

        outputs = keras.layers.Dense(1, activation='sigmoid', name='output')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='yamnet_deep_head')

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
        print(f"Total trainable parameters: {self.model.count_params():,}")

        return self.model
    
    def get_callbacks(self):
        checkpoint_dir = self.model_dir / 'checkpoints_unfreeze'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        callbacks = []
        
        # 3. الاعتماد على امتداد .keras الرسمي والحديث لتجنب تحذيرات الحفظ التقليدية
        best_model_path = self.model_dir / 'best_yamnet_finetuned_unfreeze.keras'
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
            patience=self.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(2, self.patience // 2),
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        log_path = self.model_dir.parent / 'results' / 'logs' / 'training_unfreeze.csv'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_logger = keras.callbacks.CSVLogger(str(log_path), append=False)
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(self, train_dataset, val_dataset, class_weights=None):
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING (Deep Classification Head)")
        print("=" * 60)
        
        callbacks = self.get_callbacks()
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED")
        print("=" * 60)
        print(f"Final training loss: {self.history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {self.history.history['val_loss'][-1]:.4f}")
        print(f"Best validation loss: {min(self.history.history['val_loss']):.4f}")
        
        return self.history
    
    def save_training_history(self):
        if self.history is None:
            print("No training history to save.")
            return
        
        output_path = self.model_dir.parent / 'results' / 'logs' / 'training_history_unfreeze.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(x) for x in values]
        
        with open(output_path, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"\nTraining history saved: {output_path}")

if __name__ == "__main__":
    print("ModelTrainerUnfreeze module ready")
