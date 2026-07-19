import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from pathlib import Path

class YAMNetFineTuner:
    """بناء نموذج Fine-Tuning لـ YAMNet."""
    
    def __init__(self, learning_rate=1e-4):
        self.learning_rate = learning_rate
        self.model = None
    
   def build_model(self):
        """بناء معمارية النموذج."""
        print("Loading YAMNet base model...")
        
        
        yamnet_layer = hub.KerasLayer(
            'https://tfhub.dev/google/yamnet/1',
            trainable=False, # تجميد الطبقات الأساسية
            name='yamnet_base'
        )
        
        inputs = keras.Input(shape=(16000,), dtype=tf.float32, name='audio_input')
        
        
        embeddings = yamnet_layer(inputs)[1] 
        
        embeddings_mean = tf.reduce_mean(embeddings, axis=1)
        
        x = keras.layers.Dense(256, activation='relu')(embeddings_mean)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(1, activation='sigmoid', name='drone_detector')(x)
        
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )
        
        print("Model built successfully")
        return self.model
    
    def get_model_summary(self):
        """عرض ملخص النموذج."""
        if self.model:
            return self.model.summary()
        else:
            print("Model not built yet")

if __name__ == "__main__":
    finetuner = YAMNetFineTuner()
    model = finetuner.build_model()
    finetuner.get_model_summary()
