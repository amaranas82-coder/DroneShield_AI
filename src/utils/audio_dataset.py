
import numpy as np
import librosa
from pathlib import Path
from typing import Tuple, List
import tensorflow as tf
from tqdm import tqdm


class AudioDataset:
    """    """
    
    def __init__(self, 
                 drone_dir: Path, 
                 non_drone_dir: Path,
                 sample_rate: int = 16000,
                 duration: float = 1.0):
        
        self.drone_dir = Path(drone_dir)
        self.non_drone_dir = Path(non_drone_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
        
        # تحميل قوائم الملفات
        self.drone_files = list(self.drone_dir.glob('*.wav'))
        self.non_drone_files = list(self.non_drone_dir.glob('*.wav'))
        
        print(f"Loaded {len(self.drone_files)} drone files")
        print(f"Loaded {len(self.non_drone_files)} non-drone files")
    
    def load_audio_file(self, file_path: Path) -> np.ndarray:
        try:
            # تحميل الملف الصوتي
            waveform, sr = librosa.load(
                file_path, 
                sr=self.sample_rate,
                mono=True
            )
            
            # ضبط الطول
            if len(waveform) > self.target_length:
                # قص الزيادة
                waveform = waveform[:self.target_length]
            elif len(waveform) < self.target_length:
                # ملء النقص بأصفار
                padding = self.target_length - len(waveform)
                waveform = np.pad(waveform, (0, padding), mode='constant')
            
            return waveform
            
        except Exception as e:
            print(f"Error loading {file_path.name}: {e}")
            return np.zeros(self.target_length)
    
    def prepare_dataset(self, 
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       random_state: int = 42) -> Tuple:
        
        print("\nPreparing dataset...")
        print("-" * 60)
        
        # قوائم لحفظ البيانات
        waveforms = []
        labels = []
        
        # تحميل ملفات الدرون (label = 1)
        print("\nLoading drone files...")
        for file_path in tqdm(self.drone_files, desc="Drone"):
            waveform = self.load_audio_file(file_path)
            waveforms.append(waveform)
            labels.append(1)
        
        # تحميل ملفات non-drone (label = 0)
        print("\nLoading non-drone files...")
        for file_path in tqdm(self.non_drone_files, desc="Non-Drone"):
            waveform = self.load_audio_file(file_path)
            waveforms.append(waveform)
            labels.append(0)
        
        # تحويل إلى numpy arrays
        X = np.array(waveforms, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)
        
        print(f"\nTotal samples: {len(X)}")
        print(f"Waveform shape: {X.shape}")
        
        # خلط البيانات
        np.random.seed(random_state)
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # حساب أحجام التقسيم
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # التقسيم
        X_train = X[:n_train]
        y_train = y[:n_train]
        
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        print("\nDataset split:")
        print(f"  Training:   {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Testing:    {len(X_test)} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def create_tf_dataset(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         batch_size: int = 32,
                         shuffle: bool = True) -> tf.data.Dataset:
       
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_weights(self, y_train: np.ndarray) -> dict:
         
        n_samples = len(y_train)
        n_drone = np.sum(y_train == 1)
        n_non_drone = np.sum(y_train == 0)
        
        # حساب الأوزان
        weight_drone = n_samples / (2 * n_drone)
        weight_non_drone = n_samples / (2 * n_non_drone)
        
        class_weights = {
            0: weight_non_drone,
            1: weight_drone
        }
        
        print("\nClass weights:")
        print(f"  Non-Drone (0): {weight_non_drone:.4f}")
        print(f"  Drone (1):     {weight_drone:.4f}")
        
        return class_weights


# اختبار الكود
if __name__ == "__main__":
    print("AudioDataset Module - Test Mode")
    print("=" * 60)
    
    from pathlib import Path
    
    # المسارات
    drone_dir = Path("data/processed/drone")
    non_drone_dir = Path("data/processed/non_drone")
    
    # إنشاء dataset
    dataset = AudioDataset(drone_dir, non_drone_dir)
    
    # اختبار تحميل ملف واحد
    if len(dataset.drone_files) > 0:
        test_file = dataset.drone_files[0]
        waveform = dataset.load_audio_file(test_file)
        print(f"\nTest file: {test_file.name}")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Sample values: {waveform[:10]}")
    
    print("\nAudioDataset ready for use!")
