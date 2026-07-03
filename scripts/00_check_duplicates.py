

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.config import Config
from utils.data_matcher import DataMatcher


def print_header():
    """Print program header."""
    print("\n" + "="*70)
    print("  DUPLICATE DETECTION PIPELINE")
    print("  Drone Shield AI - Data Quality Check")
    print("="*70)


def main():
    """Main function."""
    
    print_header()
    
    matcher = DataMatcher()
    
    # Step 1: Build hash database from training data
    print("\n[STEP 1/4] Building hash database from training data...")
    print("-" * 70)
    
    matcher.build_hash_database(Config.DRONE_DIR, 'train_drone')
    matcher.build_hash_database(Config.NON_DRONE_DIR, 'train_non_drone')
    
    matcher.print_summary()
    
    # Step 2: Check external test data
    print("\n[STEP 2/4] Checking external test data...")
    print("-" * 70)
    
    drone_duplicates = matcher.find_duplicates(Config.EXTERNAL_DRONE_DIR)
    non_drone_duplicates = matcher.find_duplicates(Config.EXTERNAL_NON_DRONE_DIR)
    
    all_duplicates = drone_duplicates + non_drone_duplicates
    total_duplicates = len(all_duplicates)
    
    # Step 3: Print results
    print("\n[STEP 3/4] Results Summary")
    print("-" * 70)
    
    print(f"\nDrone duplicates found:     {len(drone_duplicates)}")
    print(f"Non-Drone duplicates found: {len(non_drone_duplicates)}")
    print(f"Total duplicates:           {total_duplicates}")
    
    if total_duplicates > 0:
        print("\nWARNING: Duplicates detected!")
        print("\nDetails:")
        
        for i, dup in enumerate(all_duplicates[:10], 1):
            print(f"\n  {i}. Test:  {dup['test_file'].name}")
            print(f"     Train: {Path(dup['train_file']).name}")
            print(f"     Hash:  {dup['hash'][:16]}...")
        
        if total_duplicates > 10:
            print(f"\n  ... and {total_duplicates - 10} more")
    else:
        print("\nNo duplicates found!")
        print("External test set is clean and independent.")
    
    # Step 4: Save report and handle duplicates
    print("\n[STEP 4/4] Actions")
    print("-" * 70)
    
    if total_duplicates > 0:
        report_file = Config.RESULTS_DIR / 'logs' / 'duplicates_report.json'
        matcher.generate_report(all_duplicates, report_file)
        
        print("\nWhat would you like to do?")
        print("  1. Remove duplicates automatically")
        print("  2. Dry run (show what would be removed)")
        print("  3. Keep duplicates (exit)")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == '1':
            print("\nRemoving duplicates...")
            matcher.remove_duplicates(all_duplicates, dry_run=False)
            print("\nDuplicates removed successfully!")
        
        elif choice == '2':
            print("\nDry run mode...")
            matcher.remove_duplicates(all_duplicates, dry_run=True)
        
        else:
            print("\nKeeping duplicates. No files removed.")
    
    print("\n" + "="*70)
    print("  DUPLICATE DETECTION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
