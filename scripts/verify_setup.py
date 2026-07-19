import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import Config

def verify():
    print("\n" + "="*60)
    print("Verifying Project Setup")
    print("="*60)
    
    Config.create_all_dirs()
    print("\nAll directories created")
    
    print("\n" + "="*60)
    print("Checking Data")
    print("="*60)
    
    drone_files = list(Config.DRONE_DIR.glob('*.wav'))
    non_drone_files = list(Config.NON_DRONE_DIR.glob('*.wav'))
    
    print(f"Drone files: {len(drone_files)}")
    print(f"Non-Drone files: {len(non_drone_files)}")
    
    if len(drone_files) > 0 and len(non_drone_files) > 0:
        print("Data found and ready")
    else:
        print("\nWarning: No data found!")
        print(f"\nPlease copy your processed data to:")
        print(f"  - {Config.DRONE_DIR}")
        print(f"  - {Config.NON_DRONE_DIR}")
    
    print("\n" + "="*60)
    print("Checking Libraries")
    print("="*60)
    
    libs = {
        'numpy': 'numpy',
        'librosa': 'librosa',
        'tensorflow': 'tensorflow',
        'tensorflow_hub': 'tensorflow_hub',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib'
    }
    
    for lib, name in libs.items():
        try:
            __import__(lib)
            print(f"OK {name}")
        except ImportError:
            print(f"MISSING {name} - NOT INSTALLED")
    
    print("\n" + "="*60)
    print("Setup verification complete")
    print("="*60)

if __name__ == "__main__":
    verify()
