#!/usr/bin/env python3
"""
System Check Script for The Auto-Grader

This script checks your system capabilities and recommends the best approach
for training the Judge Model.
"""

import sys
import platform

def check_system():
    print("="*70)
    print("AUTO-GRADER SYSTEM CHECK")
    print("="*70)
    
    # 1. Operating System
    print(f"\n[SYSTEM] Operating System: {platform.system()} {platform.release()}")
    print(f"         Architecture: {platform.machine()}")
    
    is_mac = platform.system() == "Darwin"
    is_linux = platform.system() == "Linux"
    
    # 2. Python version
    python_version = sys.version.split()[0]
    print(f"\n[PYTHON] Version: {python_version}")
    
    # 3. Check PyTorch
    print(f"\n[PYTORCH] Status:")
    try:
        import torch
        print(f"   [OK] PyTorch installed: {torch.__version__}")
        
        # Check CUDA
        if torch.cuda.is_available():
            print(f"   [OK] CUDA available: {torch.version.cuda}")
            print(f"   [GPU] {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   [MEM] GPU Memory: {memory_gb:.1f} GB")
            can_train_locally = True
            training_speed = "Fast (~15-20 minutes with 4-bit quantization)"
        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"   [OK] MPS (Apple Silicon GPU) available")
            can_train_locally = True
            training_speed = "Moderate (~30-60 minutes, no quantization)"
        else:
            print(f"   [WARN] No GPU detected - CPU only")
            can_train_locally = True
            training_speed = "Very Slow (several hours, not recommended)"
    except ImportError:
        print(f"   [ERROR] PyTorch not installed")
        can_train_locally = False
        training_speed = "N/A"
    
    # 4. Check bitsandbytes
    print(f"\n[QUANTIZATION] Support:")
    try:
        import bitsandbytes
        print(f"   [OK] bitsandbytes installed: {bitsandbytes.__version__}")
        if is_linux and torch.cuda.is_available():
            print(f"   [OK] 4-bit quantization available")
            quantization_available = True
        else:
            print(f"   [WARN] 4-bit quantization requires Linux + CUDA")
            quantization_available = False
    except ImportError:
        print(f"   [WARN] bitsandbytes not installed")
        if is_mac:
            print(f"   [INFO] Note: bitsandbytes doesn't support macOS")
        quantization_available = False
    
    # 5. Recommendations
    print(f"\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    
    try:
        if is_linux and torch.cuda.is_available() and quantization_available:
            print(f"\n[OPTIMAL] YOUR SYSTEM IS READY FOR TRAINING")
            print(f"          You can train locally with 4-bit quantization")
            print(f"          Expected training time: {training_speed}")
            print(f"\n          Run: python3 src/train.py")
            
        elif is_mac and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"\n[LIMITED] YOUR SYSTEM CAN TRAIN (with limitations)")
            print(f"          Apple Silicon GPU will be used (no quantization)")
            print(f"          Expected training time: {training_speed}")
            print(f"\n          Options:")
            print(f"          1. Train locally: python3 src/train.py")
            print(f"          2. Use Google Colab (faster): notebooks/auto_grader_colab.ipynb")
            print(f"\n          Colab is recommended for faster training with quantization")
            
        else:
            print(f"\n[NOT RECOMMENDED] LOCAL TRAINING")
            print(f"                  Your system will train very slowly on CPU")
            print(f"                  Expected training time: {training_speed}")
            print(f"\n          >>> RECOMMENDED: Use Google Colab (Free GPU)")
            print(f"          1. Open notebooks/auto_grader_colab.ipynb")
            print(f"          2. Upload to Google Colab")
            print(f"          3. Run all cells (~20 minutes on free T4 GPU)")
            print(f"\n          Alternative: Try training locally (very slow)")
            print(f"          Run: python3 src/train.py")
    except:
        pass
    
    # 6. Google Colab instructions
    print(f"\n" + "="*70)
    print("GOOGLE COLAB SETUP (RECOMMENDED FOR MOST USERS)")
    print("="*70)
    print("""
1. Go to https://colab.research.google.com/
2. Click 'File' -> 'Upload notebook'
3. Upload: notebooks/auto_grader_colab.ipynb
4. Click 'Runtime' -> 'Change runtime type'
5. Select 'T4 GPU' (free tier)
6. Click 'Runtime' -> 'Run all'
7. Wait ~20 minutes for training to complete
8. Download the trained model if needed

Benefits:
[+] Free GPU access (T4)
[+] 4-bit quantization support
[+] Faster training (~15-20 min)
[+] No local setup required
[+] Pre-configured environment
""")
    
    print(f"="*70)


if __name__ == "__main__":
    try:
        check_system()
    except Exception as e:
        print(f"\n[ERROR] System check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
