import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from pathlib import Path
from tqdm import tqdm

class YAMNetExtractor:
    def __init__(self):
        print("Loading YAMNet model...")
        self.model = hub.load('https://tfhub.dev/google/yamnet/1')
        print("YAMNet loaded successfully\n")
    
    def extract_single(self, audio_file: Path) -> np.ndarray:
        y, sr = librosa.load(audio_file, sr=16000, mono=True)
        
        scores, embeddings, spectrogram = self.model(y)
        embeddings = embeddings.numpy()
        
        mean_embedding = np.mean(embeddings, axis=0)
        
        return mean_embedding
    
    def extract_batch(self, audio_files: list, labels: np.ndarray) -> tuple:
        features = []
        
        for audio_file in tqdm(audio_files, desc="Extracting features"):
            try:
                feature = self.extract_single(audio_file)
                features.append(feature)
            except Exception as e:
                print(f"Error processing {audio_file.name}: {e}")
                continue
        
        return np.array(features), labels[:len(features)]
    
    def save_features(self, features, labels, output_file: Path):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_file,
            features=features,
            labels=labels
        )
        
        print(f"\nSaved features to: {output_file}")
        print(f"  Shape: {features.shape}")
        print(f"  Labels: {len(labels)}")

if __name__ == "__main__":
    print("YAMNet Extractor module ready")
