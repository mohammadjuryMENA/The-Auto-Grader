"""
Inference Utilities for Auto-Grader Judge Model

This module provides convenient functions for running inference with the trained judge model.
"""

import torch
from pathlib import Path
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import re

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()


class JudgeInference:
    """Handles inference with the trained Judge Model."""
    
    def __init__(
        self,
        model_path: str = None,
        base_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    ):
        # Use absolute path relative to project root
        if model_path is None:
            model_path = str(PROJECT_ROOT / "models" / "judge-model")
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(
            self.model,
            self.model_path,
        )
        
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
    def create_prompt(self, prompt: str, response: str, rubric: str) -> str:
        """Create evaluation prompt."""
        return f"""You are an AI Judge. Evaluate the following response based on the rubric and provide a score from 1-5 and detailed reasoning.

**Prompt:** {prompt}

**Response:** {response}

**Rubric:** {rubric}

**Evaluation:**"""
    
    def parse_output(self, output: str) -> Tuple[int, str]:
        """Parse model output to extract score and reasoning."""
        # Look for score
        score_match = re.search(r'Score:\s*(\d+)(?:/5)?', output, re.IGNORECASE)
        
        if score_match:
            score = int(score_match.group(1))
            score = max(1, min(5, score))
        else:
            score = 3  # Default
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+)', output, re.IGNORECASE | re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            if score_match:
                reasoning = output[score_match.end():].strip()
            else:
                reasoning = output.strip()
        
        return score, reasoning
    
    def evaluate(
        self,
        prompt: str,
        response: str,
        rubric: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Evaluate a response against a rubric.
        
        Args:
            prompt: The original user prompt
            response: The AI model's response to evaluate
            rubric: The grading rubric/criteria
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Dictionary with score and reasoning
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Create prompt
        full_prompt = self.create_prompt(prompt, response, rubric)
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract model's response
        model_response = full_output[len(full_prompt):].strip()
        
        # Parse
        score, reasoning = self.parse_output(model_response)
        
        return {
            'score': score,
            'reasoning': reasoning,
            'raw_output': model_response,
        }
    
    def batch_evaluate(self, cases: list) -> list:
        """
        Evaluate multiple cases.
        
        Args:
            cases: List of dicts with 'prompt', 'response', and 'rubric' keys
        
        Returns:
            List of evaluation results
        """
        results = []
        for case in cases:
            result = self.evaluate(
                case['prompt'],
                case['response'],
                case['rubric']
            )
            results.append(result)
        return results


def quick_evaluate(prompt: str, response: str, rubric: str, model_path: str = None):
    """
    Convenience function for quick evaluation.
    
    Example:
        result = quick_evaluate(
            prompt="What is 2+2?",
            response="2+2 equals 5",
            rubric="Grade for correctness"
        )
        print(f"Score: {result['score']}/5")
        print(f"Reasoning: {result['reasoning']}")
    """
    judge = JudgeInference(model_path=model_path)
    judge.load()
    return judge.evaluate(prompt, response, rubric)


if __name__ == "__main__":
    # Example usage
    print("Testing Judge Model Inference...\n")
    
    result = quick_evaluate(
        prompt="What is the capital of France?",
        response="The capital of France is Paris.",
        rubric="Grade for Correctness: Does the response provide accurate information?"
    )
    
    print(f"Score: {result['score']}/5")
    print(f"Reasoning: {result['reasoning']}")
