import shutil
from pathlib import Path
from tqdm import tqdm
import random

print("=" * 70)
print("  Kaggle Dataset Auto-Sorter (Limited to 1000 per class)")
print("=" * 70)

raw_dir = Path("data/raw_external")
drone_dir = Path("data/external_test/drone")
non_drone_dir = Path("data/external_test/non_drone")

drone_dir.mkdir(parents=True, exist_ok=True)
non_drone_dir.mkdir(parents=True, exist_ok=True)

MAX_PER_CLASS = 1000

drone_keywords = [
    'drone', 'quadcopter', 'uav', 'membo', 'hovering', 'hoovering',
    'free_', 'u_p_', 'u_d_', 'zz_g', 'zz_l', 'd3', 'd7',
    'flyingdronesoundeffect', '15m', '10m', '30m', '5m', '20m'
]

non_drone_keywords = [
    'not_a_drone', 'exercise', 'silence', 'running_tap', 
    'pink_noise', 'electric_toothbrush'
]

all_files = list(raw_dir.rglob('*.wav'))

if len(all_files) == 0:
    print(f"\nError: No WAV files found in {raw_dir}")
    print("\nPlease:")
    print("  1. Download and extract the Kaggle dataset")
    print("  2. Move all WAV files to: data/raw_external/")
    print("  3. Run this script again")
    exit(1)

print(f"\nFound {len(all_files)} WAV files")
print("Shuffling files for random sampling...")

random.shuffle(all_files)

print("\nClassifying files...\n")

drone_files = []
non_drone_files = []

for file in tqdm(all_files, desc="Scanning files"):
    name_lower = file.name.lower()
    
    is_drone = any(kw in name_lower for kw in drone_keywords)
    is_non_drone = any(kw in name_lower for kw in non_drone_keywords)
    
    if is_drone and not is_non_drone:
        if len(drone_files) < MAX_PER_CLASS:
            drone_files.append(file)
    
    elif is_non_drone or (not is_drone):
        if len(non_drone_files) < MAX_PER_CLASS:
            non_drone_files.append(file)
    
    if len(drone_files) >= MAX_PER_CLASS and len(non_drone_files) >= MAX_PER_CLASS:
        print("\n\nReached target of 1000 files per class!")
        break

print(f"\n\nCopying {len(drone_files)} drone files...")
for file in tqdm(drone_files, desc="Drone files"):
    shutil.copy(file, drone_dir / file.name)

print(f"\nCopying {len(non_drone_files)} non-drone files...")
for file in tqdm(non_drone_files, desc="Non-drone files"):
    shutil.copy(file, non_drone_dir / file.name)

print("\n" + "=" * 70)
print("  Sorting Complete")
print("=" * 70)
print(f"\nResults:")
print(f"  Drone files:     {len(drone_files)}")
print(f"  Non-Drone files: {len(non_drone_files)}")

print(f"\nFiles saved to:")
print(f"  Drone:     {drone_dir}")
print(f"  Non-Drone: {non_drone_dir}")

if len(drone_files) < MAX_PER_CLASS or len(non_drone_files) < MAX_PER_CLASS:
    print(f"\nWarning: Could not find {MAX_PER_CLASS} files for both classes")
    print(f"Available drone files: {len(drone_files)}")
    print(f"Available non-drone files: {len(non_drone_files)}")

