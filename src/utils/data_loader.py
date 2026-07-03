from pathlib import Path
from typing import List, Tuple
import numpy as np

class DataLoader:
    def __init__(self, drone_dir: Path, non_drone_dir: Path):
        self.drone_dir = Path(drone_dir)
        self.non_drone_dir = Path(non_drone_dir)
    
    def load_file_paths(self) -> Tuple[List[Path], np.ndarray]:
        drone_files = sorted(list(self.drone_dir.glob('*.wav')))
        non_drone_files = sorted(list(self.non_drone_dir.glob('*.wav')))
        
        all_files = drone_files + non_drone_files
        
        labels = np.array(
            [1] * len(drone_files) + [0] * len(non_drone_files)
        )
        
        return all_files, labels
    
    def get_stats(self) -> dict:
        drone_count = len(list(self.drone_dir.glob('*.wav')))
        non_drone_count = len(list(self.non_drone_dir.glob('*.wav')))
        
        return {
            'drone_count': drone_count,
            'non_drone_count': non_drone_count,
            'total': drone_count + non_drone_count,
            'balance_ratio': non_drone_count / drone_count if drone_count > 0 else 0
        }

if __name__ == "__main__":
    from config import Config
    
    loader = DataLoader(Config.DRONE_DIR, Config.NON_DRONE_DIR)
    stats = loader.get_stats()
    
    print("Dataset Statistics:")
    print(f"  Drone files: {stats['drone_count']}")
    print(f"  Non-Drone files: {stats['non_drone_count']}")
    print(f"  Total: {stats['total']}")
    print(f"  Balance ratio: 1:{stats['balance_ratio']:.2f}")
