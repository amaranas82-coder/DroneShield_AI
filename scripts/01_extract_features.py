import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import Config
from utils.data_loader import DataLoader
from features.yamnet_extractor import YAMNetExtractor

def main():
    print("="*60)
    print("Feature Extraction Pipeline")
    print("="*60)
    
    Config.create_all_dirs()
    
    loader = DataLoader(Config.DRONE_DIR, Config.NON_DRONE_DIR)
    
    stats = loader.get_stats()
    print(f"\nDataset Statistics:")
    print(f"  Drone: {stats['drone_count']}")
    print(f"  Non-Drone: {stats['non_drone_count']}")
    print(f"  Total: {stats['total']}")
    
    all_files, labels = loader.load_file_paths()
    
    extractor = YAMNetExtractor()
    
    features, labels = extractor.extract_batch(all_files, labels)
    
    extractor.save_features(features, labels, Config.YAMNET_FEATURES)
    
    print("\n" + "="*60)
    print("Feature extraction complete")
    print("="*60)

if __name__ == "__main__":
    main()
