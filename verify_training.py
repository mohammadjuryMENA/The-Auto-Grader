"""
Quick verification script to check if training setup is correct.
This checks paths and dependencies without actually training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def verify_setup():
    print("="*70)
    print("TRAINING SETUP VERIFICATION")
    print("="*70)
    
    # 1. Check imports
    print("\n1. Checking imports...")
    try:
        from train import PROJECT_ROOT, JudgeModelTrainer
        print(f"   ✅ Training module imported successfully")
        print(f"   Project root: {PROJECT_ROOT}")
    except Exception as e:
        print(f"   ❌ Failed to import training module: {e}")
        return False
    
    # 2. Check dataset
    print("\n2. Checking dataset...")
    dataset_path = PROJECT_ROOT / "data" / "train_dataset.json"
    if dataset_path.exists():
        import json
        with open(dataset_path) as f:
            data = json.load(f)
        print(f"   ✅ Dataset found: {len(data)} examples")
    else:
        print(f"   ❌ Dataset not found at {dataset_path}")
        return False
    
    # 3. Check test cases
    print("\n3. Checking test cases...")
    test_cases_path = PROJECT_ROOT / "data" / "test_cases.json"
    if test_cases_path.exists():
        with open(test_cases_path) as f:
            test_cases = json.load(f)
        print(f"   ✅ Test cases found: {len(test_cases)} tests")
    else:
        print(f"   ❌ Test cases not found at {test_cases_path}")
        return False
    
    # 4. Check output directory
    print("\n4. Checking output directories...")
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    models_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    print(f"   ✅ Output directories ready")
    print(f"      Models: {models_dir}")
    print(f"      Results: {results_dir}")
    
    # 5. Check GPU availability
    print("\n5. Checking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"      GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif torch.backends.mps.is_available():
            print(f"   ✅ MPS (Apple Silicon) available")
        else:
            print(f"   ⚠️  No GPU detected - training will be slow on CPU")
            print(f"      Consider using Google Colab for free GPU access")
    except Exception as e:
        print(f"   ⚠️  Could not check GPU: {e}")
    
    print("\n" + "="*70)
    print("✅ SETUP VERIFICATION PASSED")
    print("="*70)
    print("\nYou can now run training with:")
    print("  python3 src/train.py")
    print("\nOr use the Google Colab notebook:")
    print("  notebooks/auto_grader_colab.ipynb")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = verify_setup()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
