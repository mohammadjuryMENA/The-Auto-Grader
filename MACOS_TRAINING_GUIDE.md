# Training on macOS - Important Information

## 🍎 Your Situation

You're running on **macOS**, which has the following limitations for training:

### ❌ What Doesn't Work on macOS

- **bitsandbytes library** - Only works on Linux with CUDA GPUs
- **4-bit quantization** - Requires bitsandbytes
- **CUDA GPU acceleration** - macOS doesn't support CUDA

### ✅ What Can Work on macOS

- **CPU training** - Very slow (hours instead of minutes)
- **MPS (Apple Silicon GPU)** - Moderate speed if you have M1/M2/M3 chip
- **Full precision (FP32)** - Higher memory usage, no quantization

## 🎯 Your Options

### Option 1: Google Colab (⭐ RECOMMENDED)

**Best for:** Everyone, especially macOS users

**Advantages:**

- ✅ Free GPU access (NVIDIA T4)
- ✅ 4-bit quantization support
- ✅ Fast training (~15-20 minutes)
- ✅ No local setup needed
- ✅ Pre-configured environment

**Steps:**

1. Open Google Colab: https://colab.research.google.com/
2. Upload `notebooks/auto_grader_colab.ipynb`
3. Select Runtime → Change runtime type → T4 GPU
4. Run all cells
5. Wait ~20 minutes for training
6. Download trained model (optional)

### Option 2: Local Training (Not Recommended)

**If you still want to train locally:**

```bash
# From project root
python3 src/train.py
```

**What will happen:**

- ⚠️ Training will use CPU or MPS (if Apple Silicon)
- ⚠️ No 4-bit quantization (requires more memory)
- ⚠️ Much slower than Colab
- ⚠️ May take several hours

**Memory requirements:**

- Without quantization: ~3-4GB RAM minimum
- Model size: ~1.5B parameters in FP32 = ~6GB
- You may run out of memory on machines with <8GB RAM

### Option 3: Use Pre-trained Models

**If training is not feasible:**

- Download a pre-trained judge model from HuggingFace
- Or use the base model with few-shot prompting
- Skip training and focus on evaluation/inference

## 🔧 Updated Training Script

The training script has been updated to:

- ✅ Detect macOS and adjust settings automatically
- ✅ Fall back to FP32/FP16 when quantization unavailable
- ✅ Use MPS (Apple Silicon) if available
- ✅ Provide clear warnings about limitations
- ✅ Continue training without bitsandbytes

## 📊 Performance Comparison

| Method                   | Hardware   | Time     | Memory | Quality |
| ------------------------ | ---------- | -------- | ------ | ------- |
| **Colab (GPU + 4-bit)**  | T4 GPU     | ~20 min  | ~3GB   | Best ⭐ |
| **Linux + CUDA + 4-bit** | NVIDIA GPU | ~20 min  | ~3GB   | Best ⭐ |
| **macOS MPS (no quant)** | M1/M2/M3   | ~1-2 hrs | ~6GB   | Good    |
| **macOS/Linux CPU**      | Any CPU    | 4-8 hrs  | ~6GB   | Good    |

## 🚀 Recommended Action

**For immediate results:**
Use Google Colab with the provided notebook. It's free, fast, and works perfectly.

**For understanding:**
The Auto-Grader is designed to work across platforms, but optimal performance requires:

- Linux operating system
- NVIDIA GPU with CUDA
- bitsandbytes library for quantization

Since you're on macOS, Colab is your best option for the competition submission.

## 📝 Quick Colab Guide

1. **Open Colab:** https://colab.research.google.com/
2. **Upload notebook:** File → Upload → Select `notebooks/auto_grader_colab.ipynb`
3. **Enable GPU:** Runtime → Change runtime type → Hardware accelerator: T4 GPU → Save
4. **Run training:** Runtime → Run all
5. **Monitor:** Watch the progress in each cell
6. **Results:** After ~20 minutes, you'll see evaluation metrics
7. **Download (optional):** Download trained model from Files panel

## ❓ FAQ

**Q: Can I install bitsandbytes on macOS?**
A: No, it's not officially supported. It requires Linux + CUDA.

**Q: Will training still work without bitsandbytes?**
A: Yes, but slower and uses more memory (no quantization).

**Q: Should I train locally or use Colab?**
A: **Use Colab** - it's free, fast, and designed for this use case.

**Q: Do I need to train at all?**
A: For the competition, yes. Training shows you understand the full pipeline.

**Q: What if Colab is unavailable?**
A: You can still train locally (slow), or use other cloud GPU services (Kaggle, Paperspace, etc.)

---

**Bottom Line:** Use Google Colab. The local training script will work but is not optimal for macOS users.
