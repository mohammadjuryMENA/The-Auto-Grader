"""
Quick test script to verify the Auto-Grader setup
"""

import os
import sys
import json

def test_file_structure():
    """Test that all required files exist."""
    print("Testing file structure...")
    
    required_files = [
        "README.md",
        "requirements.txt",
        "data/generate_dataset.py",
        "data/train_dataset.json",
        "data/test_cases.json",
        "src/train.py",
        "src/evaluate.py",
        "src/inference.py",
        "notebooks/auto_grader_colab.ipynb",
    ]
    
    missing = []
    for file in required_files:
        if not os.path.exists(file):
            missing.append(file)
    
    if missing:
        print(f"❌ Missing files: {', '.join(missing)}")
        return False
    else:
        print("✅ All required files present")
        return True


def test_dataset():
    """Test that dataset is properly formatted."""
    print("\nTesting dataset...")
    
    try:
        with open("data/train_dataset.json", 'r') as f:
            data = json.load(f)
        
        if len(data) < 10:
            print(f"❌ Dataset too small: {len(data)} examples")
            return False
        
        # Check format
        required_keys = ['text', 'score']
        for i, item in enumerate(data[:5]):
            for key in required_keys:
                if key not in item:
                    print(f"❌ Missing key '{key}' in example {i}")
                    return False
        
        # Check score distribution
        scores = [item['score'] for item in data]
        score_dist = {i: scores.count(i) for i in range(1, 6)}
        
        print(f"✅ Dataset valid: {len(data)} examples")
        print(f"   Score distribution: {score_dist}")
        
        # Check balance (no score should have >40% of examples)
        max_pct = max(score_dist.values()) / len(data)
        if max_pct > 0.4:
            print(f"⚠️  Warning: Dataset may be imbalanced (max {max_pct:.1%})")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset error: {e}")
        return False


def test_test_cases():
    """Test that test cases include all levels."""
    print("\nTesting test cases...")
    
    try:
        with open("data/test_cases.json", 'r') as f:
            test_cases = json.load(f)
        
        levels = set(tc['level'] for tc in test_cases)
        
        if not {1, 2, 3}.issubset(levels):
            print(f"❌ Missing challenge levels: {levels}")
            return False
        
        print(f"✅ Test cases valid: {len(test_cases)} tests")
        print(f"   Levels covered: {sorted(levels)}")
        
        level_counts = {i: len([tc for tc in test_cases if tc['level'] == i]) for i in range(1, 4)}
        print(f"   Level distribution: {level_counts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test cases error: {e}")
        return False


def test_imports():
    """Test that required packages can be imported."""
    print("\nTesting package imports...")
    
    required_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('peft', 'peft'),
        ('trl', 'trl'),
        ('datasets', 'datasets'),
    ]
    
    missing = []
    for pkg_name, import_name in required_packages:
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name}")
        except ImportError:
            print(f"  ❌ {pkg_name}")
            missing.append(pkg_name)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages installed")
        return True


def main():
    """Run all tests."""
    print("="*70)
    print("AUTO-GRADER SETUP VERIFICATION")
    print("="*70)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dataset", test_dataset),
        ("Test Cases", test_test_cases),
        ("Package Imports", test_imports),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with error: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! You're ready to train the model.")
        print("\nNext steps:")
        print("  1. Train: python src/train.py")
        print("  2. Or use Colab: notebooks/auto_grader_colab.ipynb")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
