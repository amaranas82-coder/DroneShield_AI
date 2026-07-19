
from pathlib import Path


class Config:
    """Central project configuration."""
    
    # Main paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / 'data'
    MODELS_DIR = PROJECT_ROOT / 'models'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    
    # Training data (processed)
    PROCESSED_DIR = DATA_DIR / 'processed'
    DRONE_DIR = PROCESSED_DIR / 'drone'
    NON_DRONE_DIR = PROCESSED_DIR / 'non_drone'
    
    # External test data
    EXTERNAL_TEST_DIR = DATA_DIR / 'external_test'
    EXTERNAL_DRONE_DIR = EXTERNAL_TEST_DIR / 'drone'
    EXTERNAL_NON_DRONE_DIR = EXTERNAL_TEST_DIR / 'non_drone'
    
    # Model files
    BEST_MODEL_PATH = MODELS_DIR / 'best_yamnet_finetuned.h5'
    CHECKPOINT_DIR = MODELS_DIR / 'checkpoints'
    
    # Results
    PLOTS_DIR = RESULTS_DIR / 'plots'
    LOGS_DIR = RESULTS_DIR / 'logs'
    
    # Audio parameters
    SAMPLE_RATE = 16000
    DURATION = 1.0
    
    # Deep Learning parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    
    # Open-set parameters
    CONFIDENCE_THRESHOLD = 0.85
    
    @classmethod
    def create_all_dirs(cls):
        """Create all required directories."""
        dirs = [
            cls.DATA_DIR,
            cls.PROCESSED_DIR,
            cls.DRONE_DIR,
            cls.NON_DRONE_DIR,
            cls.EXTERNAL_TEST_DIR,
            cls.EXTERNAL_DRONE_DIR,
            cls.EXTERNAL_NON_DRONE_DIR,
            cls.MODELS_DIR,
            cls.CHECKPOINT_DIR,
            cls.RESULTS_DIR,
            cls.PLOTS_DIR,
            cls.LOGS_DIR
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def verify_data(cls):
        """Verify data existence."""
        drone_files = list(cls.DRONE_DIR.glob('*.wav'))
        non_drone_files = list(cls.NON_DRONE_DIR.glob('*.wav'))
        
        print(f"Training Data:")
        print(f"  Drone files: {len(drone_files)}")
        print(f"  Non-Drone files: {len(non_drone_files)}")
        
        ext_drone = list(cls.EXTERNAL_DRONE_DIR.glob('*.wav'))
        ext_non_drone = list(cls.EXTERNAL_NON_DRONE_DIR.glob('*.wav'))
        
        print(f"\nExternal Test Data:")
        print(f"  Drone files: {len(ext_drone)}")
        print(f"  Non-Drone files: {len(ext_non_drone)}")
        
        return len(drone_files) > 0 and len(non_drone_files) > 0


if __name__ == "__main__":
    Config.create_all_dirs()
    print("Configuration initialized successfully!")
    Config.verify_data()
