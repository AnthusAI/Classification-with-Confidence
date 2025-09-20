#!/usr/bin/env python3
"""
Fine-Tune Llama 3.1 for Sentiment Classification with Better Confidence Calibration

This script uses LoRA (Low-Rank Adaptation) to efficiently fine-tune Llama 3.1
on our sentiment classification dataset for improved accuracy and confidence calibration.

Usage:
    python fine_tune_model.py

Requirements:
    pip install peft bitsandbytes datasets accelerate wandb
"""

import os
import torch
import json
from typing import Dict, List, Any, Optional

# Set environment variables to prevent warnings and issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer fork warnings
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb completely
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset
import warnings
warnings.filterwarnings("ignore")


class SentimentFineTuner:
    """
    Fine-tune Llama 3.1 for sentiment classification with improved confidence calibration.
    """

    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 output_dir: str = "./fine_tuned_sentiment_model"):
        """
        Initialize the fine-tuner.

        Args:
            model_name: HuggingFace model identifier
            output_dir: Directory to save the fine-tuned model
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _get_best_device(self) -> str:
        """Determine the best available device for training."""
        if torch.backends.mps.is_available():
            try:
                # Test MPS with a small tensor
                test_tensor = torch.randn(10, 10).to('mps')
                del test_tensor
                return "mps"
            except Exception as e:
                print(f"⚠️  MPS available but not working: {e}")
        
        if torch.cuda.is_available():
            try:
                # Test CUDA with a small tensor
                test_tensor = torch.randn(10, 10).to('cuda')
                del test_tensor
                return "cuda"
            except Exception as e:
                print(f"⚠️  CUDA available but not working: {e}")
        
        return "cpu"
    
    def _report_device_info(self, device: str):
        """Report information about the device being used."""
        if device == "mps":
            print("🚀 Using Apple Silicon GPU (MPS) for training!")
            if self.model is not None:
                print(f"Model device: {next(self.model.parameters()).device}")
        elif device == "cuda":
            print(f"🚀 Using NVIDIA GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️  Using CPU - training will be slower")

    def load_model_and_tokenizer(self):
        """Load the base model and tokenizer."""
        print(f"Loading {self.model_name}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (no quantization on Mac/MPS)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )

        # Set device with graceful fallback
        device = self._get_best_device()
        
        try:
            self.model = self.model.to(device)
            print(f"✅ Model loaded successfully on {device}!")
            
            # Verify and report GPU usage
            self._report_device_info(device)
            
        except Exception as e:
            print(f"⚠️  Failed to load model on {device}: {e}")
            if device != "cpu":
                print("🔄 Falling back to CPU...")
                device = "cpu"
                self.model = self.model.to(device)
                print("✅ Model loaded successfully on CPU!")
                self._report_device_info(device)
            else:
                raise

    def setup_lora_config(self) -> LoraConfig:
        """Setup LoRA configuration for efficient fine-tuning."""

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,  # Rank - higher values give more capacity but use more memory
            lora_alpha=16,  # LoRA scaling parameter
            lora_dropout=0.1,  # Dropout for regularization
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj"      # MLP layers
            ],
            bias="none",
            inference_mode=False,
        )

        return lora_config

    def apply_lora(self):
        """Apply LoRA to the model."""
        print("Applying LoRA configuration...")

        lora_config = self.setup_lora_config()
        self.model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())

        print(f"Trainable parameters: {trainable_params:,}")
        print(f"All parameters: {all_params:,}")
        print(f"Trainable percentage: {100 * trainable_params / all_params:.2f}%")

    def create_fine_tuning_dataset(self) -> Dataset:
        """Convert our dataset to HuggingFace format for instruction tuning."""
        print("Preparing fine-tuning dataset...")

        from sentiment_datasets import get_test_sets

        # Get all examples
        examples = get_test_sets()['all']
        print(f"Total examples: {len(examples)}")

        # Format for instruction tuning
        formatted_data = []

        for example in examples:
            text = example['text']
            label = example['expected']
            category = example.get('category', 'unknown')

            # Create instruction-response pairs
            instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nClassify the sentiment of this text as either 'positive' or 'negative':\n\n\"{text}\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            response = f"{label}<|eot_id|>"

            # Combine instruction and response for training
            full_text = instruction + response

            formatted_data.append({
                'text': full_text,
                'input_text': text,
                'label': label,
                'category': category,
                'instruction': instruction,
                'response': response
            })

        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(formatted_data)

        # Split into train/validation (80/20)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)

        print(f"Training examples: {len(dataset['train'])}")
        print(f"Validation examples: {len(dataset['test'])}")

        return dataset

    def tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenize the dataset for training."""
        print("Tokenizing dataset...")

        def tokenize_function(examples):
            # Tokenize the full text (instruction + response)
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,  # We'll pad in the data collator
                max_length=512,  # Reasonable max length for sentiment classification
                return_tensors=None
            )

            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].copy()

            return tokenized

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )

        return tokenized_dataset

    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training configuration."""

        training_args = TrainingArguments(
            output_dir=self.output_dir,

            # Training schedule
            num_train_epochs=3,
            max_steps=1000,  # Limit steps to prevent overfitting

            # Batch sizes
            per_device_train_batch_size=2,  # Small batch size for memory efficiency
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16

            # Learning rate and optimization
            learning_rate=5e-4,
            weight_decay=0.01,
            warmup_steps=100,

            # Memory and precision (disable fp16 on MPS)
            fp16=torch.cuda.is_available(),  # Only use fp16 on CUDA, not MPS
            dataloader_pin_memory=False,

            # Logging and evaluation
            logging_steps=25,
            eval_strategy="steps",
            eval_steps=100,
            save_steps=200,

            # Model saving
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Other settings
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Prevent multiprocessing issues
            report_to=[],  # Disable all reporting (wandb, tensorboard, etc.)
        )

        return training_args

    def fine_tune(self):
        """Main fine-tuning function."""
        print("🚀 Starting fine-tuning process...")

        # Load model and setup LoRA
        self.load_model_and_tokenizer()
        self.apply_lora()

        # Create and tokenize dataset
        dataset = self.create_fine_tuning_dataset()
        tokenized_dataset = self.tokenize_dataset(dataset)

        # Setup training
        training_args = self.setup_training_arguments()

        # Data collator for padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        print("Starting training...")
        
        # Monitor GPU usage during training
        device = next(self.model.parameters()).device
        if str(device).startswith('mps'):
            print("🔥 Training on Apple Silicon GPU - monitoring memory usage...")
        
        # Train the model
        trainer.train()

        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"✅ Fine-tuning completed! Model saved to {self.output_dir}")

        # Save training info
        training_info = {
            'base_model': self.model_name,
            'training_examples': len(tokenized_dataset['train']),
            'validation_examples': len(tokenized_dataset['test']),
            'training_args': training_args.to_dict(),
            'lora_config': self.setup_lora_config().to_dict()
        }

        with open(os.path.join(self.output_dir, 'training_info.json'), 'w') as f:
            json.dump(training_info, f, indent=2)

        return trainer


class FineTunedSentimentClassifier:
    """
    Load and use the fine-tuned sentiment classifier.
    """

    def __init__(self,
                 base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 fine_tuned_path: str = "./fine_tuned_sentiment_model"):
        """
        Initialize the fine-tuned classifier.

        Args:
            base_model_name: Original base model name
            fine_tuned_path: Path to the fine-tuned model
        """
        self.base_model_name = base_model_name
        self.fine_tuned_path = fine_tuned_path
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the fine-tuned model."""
        if not os.path.exists(self.fine_tuned_path):
            raise FileNotFoundError(f"Fine-tuned model not found at {self.fine_tuned_path}")

        print(f"Loading fine-tuned model from {self.fine_tuned_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_path)

        # Load base model (no quantization on Mac/MPS)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16
        )

        # Load LoRA weights
        self.model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)

        print("✅ Fine-tuned model loaded successfully!")

    def classify_single(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text with the fine-tuned model.

        Args:
            text: Text to classify

        Returns:
            Dictionary with classification results
        """
        if self.model is None:
            self.load_model()

        # Create instruction prompt
        instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nClassify the sentiment of this text as either 'positive' or 'negative':\n\n\"{text}\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        # Tokenize
        inputs = self.tokenizer(instruction, return_tensors="pt")

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response with logits
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,  # Deterministic for consistent results
                temperature=0.0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode response
        response_ids = outputs.sequences[0][len(inputs['input_ids'][0]):]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip().lower()

        # Extract logits for confidence calculation
        if outputs.scores:
            logits = outputs.scores[0][0]  # First generated token
            probs = torch.softmax(logits, dim=-1)

            # Get probabilities for positive/negative tokens
            pos_tokens = ["positive", "pos"]
            neg_tokens = ["negative", "neg"]

            pos_prob = 0.0
            neg_prob = 0.0

            for token in pos_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    pos_prob += probs[token_ids[0]].item()

            for token in neg_tokens:
                token_ids = self.tokenizer.encode(token, add_special_tokens=False)
                if token_ids:
                    neg_prob += probs[token_ids[0]].item()

            # Normalize
            total_prob = pos_prob + neg_prob
            if total_prob > 0:
                pos_prob /= total_prob
                neg_prob /= total_prob

            # Determine prediction and confidence
            if pos_prob > neg_prob:
                prediction = "positive"
                confidence = pos_prob
            else:
                prediction = "negative"
                confidence = neg_prob

        else:
            # Fallback: parse response text
            if "positive" in response:
                prediction = "positive"
                confidence = 0.5  # Unknown confidence
            elif "negative" in response:
                prediction = "negative"
                confidence = 0.5  # Unknown confidence
            else:
                prediction = "positive"  # Default
                confidence = 0.5

        return {
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'raw_response': response,
            'positive_prob': pos_prob if 'pos_prob' in locals() else 0.5,
            'negative_prob': neg_prob if 'neg_prob' in locals() else 0.5,
            'model_type': 'fine_tuned'
        }


def main():
    """Run the fine-tuning process."""
    print("🎯 Fine-Tuning Llama 3.1 for Sentiment Classification")
    print("=" * 60)
    
    # Check for required dependencies
    try:
        import peft
        import bitsandbytes
        import datasets
        import accelerate
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("Please install fine-tuning dependencies:")
        print("pip install peft bitsandbytes datasets accelerate")
        return

    # Check available compute devices
    print("🔍 Checking available compute devices...")
    if torch.backends.mps.is_available():
        print("✅ Apple Silicon GPU (MPS) available")
    if torch.cuda.is_available():
        print(f"✅ NVIDIA GPU available: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("⚠️  No GPU available - training will be slow on CPU")

    # Initialize fine-tuner
    fine_tuner = SentimentFineTuner()

    # Run fine-tuning
    try:
        trainer = fine_tuner.fine_tune()

        print("\n🎉 Fine-tuning completed successfully!")
        print(f"Model saved to: {fine_tuner.output_dir}")

        # Test the fine-tuned model
        print("\n🧪 Testing fine-tuned model...")
        classifier = FineTunedSentimentClassifier(fine_tuned_path=fine_tuner.output_dir)

        test_examples = [
            "I absolutely love this amazing movie!",
            "This is terrible and boring.",
            "Best worst thing ever",
            "Loving the hate"
        ]

        for text in test_examples:
            result = classifier.classify_single(text)
            print(f"'{text}' → {result['prediction']} ({result['confidence']:.3f})")

    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()