

import hashlib
from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
import json


class DataMatcher:
    """
    Audio file duplicate detector using MD5 hashing.
    
    This class builds a hash database of training files and checks
    test files against it to find duplicates.
    
    Attributes:
        hashes (dict): Database storing {hash: file_info}
    """
    
    def __init__(self):
        """Initialize empty hash database."""
        self.hashes = {}
    
    def compute_audio_hash(self, audio_file: Path) -> str:
        """
        Compute MD5 hash of audio file content.
        
        Process:
        --------
        1. Load audio file (16kHz, mono)
        2. Convert waveform to bytes
        3. Compute MD5 hash
        
        Parameters:
        -----------
        audio_file : Path
            Path to audio file
        
        Returns:
        --------
        str or None
            32-character hex hash or None if failed
        
        Example:
        --------
        hash_value = matcher.compute_audio_hash("drone_001.wav")
        """
        try:
            # Load audio at 16kHz (same as YAMNet)
            y, sr = librosa.load(audio_file, sr=16000, mono=True)
            
            # Convert waveform to binary data
            audio_bytes = y.tobytes()
            
            # Compute MD5 hash
            file_hash = hashlib.md5(audio_bytes).hexdigest()
            
            return file_hash
            
        except Exception as e:
            print(f"Warning: Error processing {audio_file.name}: {e}")
            return None
    
    def build_hash_database(self, data_dir: Path, label: str):
        """
        Build hash database from entire directory.
        
        Parameters:
        -----------
        data_dir : Path
            Path to data directory
        label : str
            Label for this directory (e.g., 'train_drone')
        
        Side Effects:
        -------------
        Populates self.hashes with new entries
        """
        files = list(data_dir.glob('*.wav'))
        
        if len(files) == 0:
            print(f"Warning: No files found in {data_dir}")
            return
        
        print(f"\nScanning: {label}")
        
        # Process files with progress bar
        for audio_file in tqdm(files, desc=f"  Hashing", unit="file"):
            file_hash = self.compute_audio_hash(audio_file)
            
            if file_hash:
                # Store file info in database
                self.hashes[file_hash] = {
                    'path': str(audio_file),
                    'name': audio_file.name,
                    'label': label
                }
    
    def find_duplicates(self, test_dir: Path) -> list:
        """
        Find duplicate files in test directory.
        
        Parameters:
        -----------
        test_dir : Path
            Path to test directory
        
        Returns:
        --------
        list of dict
            List of duplicates, each containing:
            - test_file: Path object
            - train_file: str path
            - train_label: str
            - hash: str
        """
        test_files = list(test_dir.glob('*.wav'))
        duplicates = []
        
        if len(test_files) == 0:
            print(f"Warning: No test files in {test_dir}")
            return duplicates
        
        print(f"\nChecking: {test_dir.name}")
        
        for test_file in tqdm(test_files, desc="  Matching", unit="file"):
            test_hash = self.compute_audio_hash(test_file)
            
            # Check if hash exists in training database
            if test_hash and test_hash in self.hashes:
                duplicates.append({
                    'test_file': test_file,
                    'train_file': self.hashes[test_hash]['path'],
                    'train_label': self.hashes[test_hash]['label'],
                    'hash': test_hash
                })
        
        return duplicates
    
    def remove_duplicates(self, duplicates: list, dry_run: bool = False):
        """
        Remove duplicate files.
        
        Parameters:
        -----------
        duplicates : list
            List of duplicates from find_duplicates()
        dry_run : bool
            If True, only print without deleting
        """
        if len(duplicates) == 0:
            print("\nNo duplicates to remove")
            return
        
        print(f"\nRemoving {len(duplicates)} duplicates...")
        
        for dup in duplicates:
            test_file = dup['test_file']
            train_file = dup['train_file']
            
            if dry_run:
                print(f"  [DRY RUN] Would remove: {test_file.name}")
                print(f"            (matches: {Path(train_file).name})")
            else:
                try:
                    test_file.unlink()
                    print(f"  Removed: {test_file.name}")
                except Exception as e:
                    print(f"  Failed to remove {test_file.name}: {e}")
    
    def generate_report(self, duplicates: list, output_file: Path = None):
        """
        Generate detailed report of duplicates.
        
        Parameters:
        -----------
        duplicates : list
            List of duplicate files
        output_file : Path, optional
            Path to save JSON report
        
        Returns:
        --------
        dict
            Report dictionary
        """
        report = {
            'total_duplicates': len(duplicates),
            'duplicates': []
        }
        
        for dup in duplicates:
            report['duplicates'].append({
                'test_file': str(dup['test_file']),
                'train_file': dup['train_file'],
                'train_label': dup['train_label'],
                'hash': dup['hash']
            })
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved: {output_file}")
        
        return report
    
    def print_summary(self):
        """Print database summary statistics."""
        print("\n" + "="*60)
        print("Hash Database Summary")
        print("="*60)
        print(f"Total unique files indexed: {len(self.hashes)}")
        
        # Statistics by label
        labels = {}
        for info in self.hashes.values():
            label = info['label']
            labels[label] = labels.get(label, 0) + 1
        
        for label, count in labels.items():
            print(f"  {label}: {count}")
        print("="*60)


if __name__ == "__main__":
    print("DataMatcher Module - Test Mode")
    print("="*60)
    
    # Create matcher instance
    matcher = DataMatcher()
    
    # Test on single file if available
    from pathlib import Path
    
    test_file = Path("data/processed/drone").glob("*.wav")
    test_file = next(test_file, None)
    
    if test_file:
        print(f"\nTesting on: {test_file.name}")
        hash_value = matcher.compute_audio_hash(test_file)
        print(f"MD5 Hash: {hash_value}")
    else:
        print("\nNo test files found. Module is ready to use.")
