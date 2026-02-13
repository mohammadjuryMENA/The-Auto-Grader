# Quick Start Guide

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/YOUR_USERNAME/The-Auto-Grader.git
cd The-Auto-Grader
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Training Dataset

```bash
python data/generate_dataset.py
```

This creates:

- `data/train_dataset.json`: 50+ training examples with balanced score distribution
- `data/test_cases.json`: Test cases for all 3 challenge levels

### 2. Train the Model

**Option A: Local Training (requires GPU)**

```bash
python src/train.py
```

**Option B: Google Colab (Free GPU)**

- Open `notebooks/auto_grader_colab.ipynb` in Google Colab
- Run all cells
- Training takes ~15-20 minutes on T4 GPU

### 3. Evaluate the Model

```bash
python src/evaluate.py
```

To evaluate the base model (before training) for comparison:

```bash
python src/evaluate.py --base-model
```

### 4. Use for Inference

```python
from src.inference import JudgeInference

# Initialize
judge = JudgeInference(model_path="models/judge-model")
judge.load()

# Evaluate
result = judge.evaluate(
    prompt="What is 2+2?",
    response="2+2 equals 5.",
    rubric="Grade for Correctness: Is the answer correct?"
)

print(f"Score: {result['score']}/5")
print(f"Reasoning: {result['reasoning']}")
```

## Project Structure

```
The-Auto-Grader/
├── data/
│   ├── generate_dataset.py      # Creates training data
│   ├── train_dataset.json       # Training examples (generated)
│   └── test_cases.json          # Test cases (generated)
├── src/
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── inference.py             # Inference utilities
├── notebooks/
│   └── auto_grader_colab.ipynb  # Google Colab notebook
├── models/
│   └── judge-model/             # Trained model (created after training)
├── results/
│   └── evaluation_results.json  # Evaluation results (created after eval)
└── requirements.txt
```

## The Three Challenge Levels

### Level 1: The Basics

Test basic correctness detection:

- Math errors (e.g., "2+2=5")
- Factual hallucinations (e.g., "Elon Musk is CEO of Apple")

### Level 2: The Stress Test

Test context-aware grading:

- Over-refusal trap (refuses to help with "kill process in Linux")

### Level 3: The Bonus Challenge

Test robustness against adversarial inputs:

- Jailbreak resistance (prompt injection attacks)

## Expected Results

After training, you should see:

- **Exact Match Accuracy**: 70-100%
- **Within-1 Accuracy**: 90-100%
- **Level 1 Performance**: 100% (basic tests)
- **Level 2 Performance**: 80%+ (context-aware)
- **Level 3 Performance**: 80%+ (jailbreak resistance)

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter OOM errors during training:

1. Reduce `per_device_train_batch_size` in `src/train.py`
2. Increase `gradient_accumulation_steps`
3. Reduce `max_seq_length` from 1024 to 512

### Model Not Loading

Ensure you have enough disk space:

- Model download: ~3GB
- Training artifacts: ~2GB

### Low Accuracy

If accuracy is low:

1. Check dataset balance (score distribution should be roughly equal)
2. Increase `num_train_epochs` from 3 to 5
3. Adjust learning rate

## Advanced Usage

### Custom Dataset

To use your own grading examples:

1. Edit `data/generate_dataset.py`
2. Add examples in the appropriate score category
3. Regenerate dataset: `python data/generate_dataset.py`

### Hyperparameter Tuning

Edit training parameters in `src/train.py`:

```python
trainer = JudgeModelTrainer(
    num_train_epochs=5,          # Increase training
    learning_rate=1e-4,          # Lower learning rate
    per_device_train_batch_size=4,  # Larger batch
)
```

### Different Base Model

To use a different model (e.g., Llama-3.2-1B):

```python
trainer = JudgeModelTrainer(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
)
```

## Citation

If you use this project, please cite:

```
The Auto-Grader: Training Specialized Judge Models
MENA Devs Competition - Track B
GitHub: https://github.com/YOUR_USERNAME/The-Auto-Grader
```

## License

MIT License - See LICENSE file for details
