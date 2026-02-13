"""
Training Script for Auto-Grader Judge Model

This script fine-tunes a small language model (<3B parameters) using Supervised Fine-Tuning (SFT)
to act as a judge that can evaluate AI responses based on rubrics.

Key Features:
- Uses Qwen-2.5-1.5B-Instruct (1.5B parameters)
- 4-bit quantization for efficient training on limited hardware
- Gradient checkpointing to reduce memory usage
- LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
"""

import os
import json
import sys
import platform
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Check for bitsandbytes availability
try:
    from transformers import BitsAndBytesConfig
    import bitsandbytes
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    logger.warning("bitsandbytes not available - 4-bit quantization will be disabled")


class JudgeModelTrainer:
    """Handles the complete training pipeline for the Judge Model."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        dataset_path: str = None,
        output_dir: str = None,
        max_seq_length: int = 1024,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        use_4bit: bool = True,
    ):
        self.model_name = model_name
        # Use absolute paths relative to project root
        if dataset_path is None:
            dataset_path = str(PROJECT_ROOT / "data" / "train_dataset.json")
        self.dataset_path = dataset_path
        if output_dir is None:
            output_dir = str(PROJECT_ROOT / "models" / "judge-model")
        self.output_dir = output_dir
        self.max_seq_length = max_seq_length
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.use_4bit = use_4bit
        
        self.model = None
        self.tokenizer = None
        self.dataset = None
    
    def load_dataset(self):
        """Load and prepare the training dataset."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to HuggingFace Dataset
        self.dataset = Dataset.from_list(data)
        
        logger.info(f"Loaded {len(self.dataset)} training examples")
        
        # Verify class balance
        score_distribution = {}
        for item in data:
            score = item['score']
            score_distribution[score] = score_distribution.get(score, 0) + 1
        
        logger.info("Score distribution:")
        for score in sorted(score_distribution.keys()):
            logger.info(f"  Score {score}: {score_distribution[score]} examples")
    
    def load_model_and_tokenizer(self):
        """Load the base model with quantization and prepare for training."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Check platform and adjust settings
        is_mac = platform.system() == "Darwin"
        has_cuda = torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        # Determine if we can use 4-bit quantization
        can_use_4bit = self.use_4bit and HAS_BITSANDBYTES and has_cuda
        
        if self.use_4bit and not can_use_4bit:
            if not HAS_BITSANDBYTES:
                logger.warning("⚠️  4-bit quantization disabled: bitsandbytes not installed")
                logger.warning("   Install with: pip install bitsandbytes (Linux/CUDA only)")
            elif not has_cuda:
                logger.warning("⚠️  4-bit quantization disabled: CUDA GPU not available")
            logger.warning("   Falling back to FP16 training (higher memory usage)")
            logger.warning("   For GPU training, use Google Colab: notebooks/auto_grader_colab.ipynb")
        
        # Configure quantization if available
        bnb_config = None
        if can_use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logger.info("✅ Using 4-bit quantization for memory efficiency")
        
        # Determine dtype and device map
        if is_mac and has_mps:
            # MPS (Apple Silicon) doesn't support float16 well
            model_dtype = torch.float32
            device_map = None  # MPS doesn't work with device_map="auto"
            logger.info("🍎 Detected Apple Silicon - using MPS with FP32")
        elif has_cuda:
            model_dtype = torch.float16
            device_map = "auto"
            logger.info("🚀 Using CUDA GPU")
        else:
            model_dtype = torch.float32
            device_map = None
            logger.warning("⚠️  Training on CPU - this will be very slow!")
            logger.warning("   Consider using Google Colab for free GPU access")
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
        
        # Move to MPS if available and not using device_map
        if is_mac and has_mps and device_map is None:
            self.model = self.model.to('mps')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )
        
        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        logger.info("Model and tokenizer loaded successfully")
    
    def prepare_model_for_training(self):
        """Apply LoRA and prepare model for k-bit training."""
        logger.info("Preparing model for training with LoRA")
        
        # Prepare model for k-bit training (enables gradient checkpointing)
        can_use_4bit = self.use_4bit and HAS_BITSANDBYTES and torch.cuda.is_available()
        if can_use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        else:
            # Enable gradient checkpointing for non-quantized models
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
        
        # LoRA configuration for parameter-efficient fine-tuning
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    def train(self):
        """Execute the training loop."""
        logger.info("Starting training...")
        
        # Determine optimizer and precision settings
        can_use_4bit = self.use_4bit and HAS_BITSANDBYTES and torch.cuda.is_available()
        has_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        if can_use_4bit:
            use_fp16 = True
            optimizer = "paged_adamw_8bit"
        elif torch.cuda.is_available():
            use_fp16 = True
            optimizer = "adamw_torch"
        elif has_mps:
            # MPS doesn't support fp16 training well
            use_fp16 = False
            optimizer = "adamw_torch"
        else:
            use_fp16 = False
            optimizer = "adamw_torch"
        
        # Training configuration
        # Calculate warmup steps (10% of total steps)
        total_steps = (len(self.dataset) // (self.per_device_train_batch_size * self.gradient_accumulation_steps)) * self.num_train_epochs
        warmup_steps = int(0.1 * total_steps)
        
        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=use_fp16,
            optim=optimizer,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
            packing=False,  # Don't pack multiple examples together
        )
        
        # Initialize trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            processing_class=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        # Save final model
        logger.info(f"Saving final model to {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info("Training completed successfully!")
    
    def run_full_pipeline(self):
        """Execute the complete training pipeline."""
        logger.info("="*80)
        logger.info("STARTING AUTO-GRADER JUDGE MODEL TRAINING PIPELINE")
        logger.info("="*80)
        
        self.load_dataset()
        self.load_model_and_tokenizer()
        self.prepare_model_for_training()
        self.train()
        
        logger.info("="*80)
        logger.info("TRAINING PIPELINE COMPLETED")
        logger.info("="*80)


def main():
    """Main entry point for training."""
    # Use absolute paths
    dataset_path = PROJECT_ROOT / "data" / "train_dataset.json"
    output_dir = PROJECT_ROOT / "models" / "judge-model"
    
    # Check if dataset exists
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.error("Please run 'python data/generate_dataset.py' first")
        return
    
    # Initialize and run trainer
    trainer = JudgeModelTrainer(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        max_seq_length=1024,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        use_4bit=True,
    )
    
    trainer.run_full_pipeline()


if __name__ == "__main__":
    main()
