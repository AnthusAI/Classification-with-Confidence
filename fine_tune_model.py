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
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # COMMENTED OUT - we want to see training progress!
import warnings
warnings.filterwarnings("ignore")

# Disable bitsandbytes on MPS to prevent compatibility issues
import torch
if torch.backends.mps.is_available():
    import os
    os.environ["BITSANDBYTES_NOWELCOME"] = "1"
    # Disable bitsandbytes optimizers on MPS
    import warnings
    warnings.filterwarnings("ignore", message=".*bitsandbytes.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer fork warnings
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb completely
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # COMMENTED OUT - we want training progress!
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
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
                print(f"WARNING:  MPS available but not working: {e}")
        
        if torch.cuda.is_available():
            try:
                # Test CUDA with a small tensor
                test_tensor = torch.randn(10, 10).to('cuda')
                del test_tensor
                return "cuda"
            except Exception as e:
                print(f"WARNING:  CUDA available but not working: {e}")
        
        return "cpu"
    
    def _report_device_info(self, device: str):
        """Report information about the device being used."""
        if device == "mps":
            print("Starting Using Apple Silicon GPU (MPS) for training!")
            if self.model is not None:
                print(f"Model device: {next(self.model.parameters()).device}")
        elif device == "cuda":
            print(f"Starting Using NVIDIA GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("WARNING:  Using CPU - training will be slower")

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
            print(f"SUCCESS: Model loaded successfully on {device}!")
            
            # Verify and report GPU usage
            self._report_device_info(device)
            
        except Exception as e:
            print(f"WARNING:  Failed to load model on {device}: {e}")
            if device != "cpu":
                print("🔄 Falling back to CPU...")
                device = "cpu"
                self.model = self.model.to(device)
                print("SUCCESS: Model loaded successfully on CPU!")
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

        from dataset_loader import DatasetLoader

        # Get all examples from new dataset
        loader = DatasetLoader()
        all_examples = loader.load_all()
        print(f"Total examples: {len(all_examples)}")
        
        # CRITICAL: Split FIRST into train/test (80/20), then format only training data
        import random
        random.seed(42)  # For reproducibility
        random.shuffle(all_examples)
        print("✅ Dataset shuffled for balanced train/test split")
        
        # Split into train (80%) and test (20%) sets
        split_idx = int(0.8 * len(all_examples))
        train_examples = all_examples[:split_idx]
        test_examples = all_examples[split_idx:]
        
        print(f"Training examples: {len(train_examples)}")
        print(f"Test examples (held out): {len(test_examples)}")
        
        # CRITICAL: Save the test set BEFORE formatting for proper evaluation
        import json
        os.makedirs('fine_tuned_sentiment_model', exist_ok=True)
        with open('fine_tuned_sentiment_model/test_set.json', 'w') as f:
            json.dump(test_examples, f, indent=2)
        print(f"✅ Test set saved to fine_tuned_sentiment_model/test_set.json ({len(test_examples)} examples)")

        # Format ONLY the training data for instruction tuning
        formatted_data = []

        for example in train_examples:
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

        # Convert training data to HuggingFace dataset (NO FURTHER SPLITTING!)
        train_dataset = Dataset.from_list(formatted_data)
        
        # Format test examples for validation during training
        test_formatted_data = []
        for example in test_examples:
            text = example['text']
            label = example['expected']
            category = example.get('category', 'unknown')

            # Create instruction-response pairs for test set
            instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nClassify the sentiment of this text as either 'positive' or 'negative':\n\n\"{text}\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            response = f"{label}<|eot_id|>"
            full_text = instruction + response

            test_formatted_data.append({
                'text': full_text,
                'input_text': text,
                'label': label,
                'category': category,
                'instruction': instruction,
                'response': response
            })
        
        test_dataset = Dataset.from_list(test_formatted_data)
        
        # Create dataset dict with train/test split (NO DOUBLE SPLITTING!)
        dataset = {
            'train': train_dataset,
            'test': test_dataset
        }

        print(f"✅ Training examples: {len(dataset['train'])} (all 8,000 used for training)")
        print(f"✅ Validation examples: {len(dataset['test'])} (held-out 2,000 test set)")
        print("📊 Single 80/20 split: 8,000 train + 2,000 validation")

        return dataset

    def tokenize_dataset(self, dataset) -> dict:
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

        # Apply tokenization to both train and test datasets
        tokenized_dataset = {
            'train': dataset['train'].map(
                tokenize_function,
                batched=True,
                remove_columns=dataset['train'].column_names
            ),
            'test': dataset['test'].map(
                tokenize_function,
                batched=True,
                remove_columns=dataset['test'].column_names
            )
        }

        return tokenized_dataset

    def setup_training_arguments(self) -> TrainingArguments:
        """Setup training configuration."""

        training_args = TrainingArguments(
            output_dir=self.output_dir,

            # Training schedule - AGGRESSIVE anti-overfitting
            num_train_epochs=1,     # Reduced from 2 - stop much earlier
            max_steps=200,          # Reduced from 400 - half the training

            # Batch sizes
            per_device_train_batch_size=2,  # Small batch size for memory efficiency
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16

            # Learning rate and optimization
            # Use standard optimizer on MPS, bitsandbytes on CUDA
            optim="adamw_torch" if torch.backends.mps.is_available() else "adamw_bnb_8bit",
            learning_rate=1e-4,     # Much lower learning rate (was 2e-4)
            weight_decay=0.05,      # Much stronger regularization (was 0.02)
            warmup_steps=100,

            # Memory and precision (disable fp16 on MPS)
            fp16=torch.cuda.is_available(),  # Only use fp16 on CUDA, not MPS
            dataloader_pin_memory=False,

            # Logging and evaluation - MORE VERBOSE
            logging_steps=10,  # Log every 10 steps for better visibility
            eval_strategy="steps",
            eval_steps=25,  # Very frequent evaluation to catch overfitting early
            save_steps=50,  # Very frequent saves
            logging_first_step=True,  # Log the first step
            log_level="info",  # Ensure info-level logging
            logging_nan_inf_filter=False,  # Don't filter out NaN/inf logs

            # Model saving
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,


            # Other settings
            remove_unused_columns=False,
            dataloader_num_workers=0,  # Prevent multiprocessing issues
            report_to=[],  # Disable external reporting but keep console output
            disable_tqdm=False,  # ENABLE progress bars
            logging_dir=None,  # No tensorboard logging
        )

        return training_args

    def fine_tune(self):
        """Main fine-tuning function."""
        print("Starting Starting fine-tuning process...")

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
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=2,      # Stop after 2 bad evaluations (was 3)
            early_stopping_threshold=0.005  # More sensitive threshold (was 0.01)
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['test'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=[early_stopping],
        )

        print("Starting training...")
        
        # Monitor GPU usage during training
        device = next(self.model.parameters()).device
        if str(device).startswith('mps'):
            print("Training on Apple Silicon GPU - monitoring memory usage...")
        
        # Enable detailed logging
        import logging
        logging.basicConfig(level=logging.INFO)
        
        print("🚀 STARTING TRAINING...")
        print(f"📊 Training steps: {len(tokenized_dataset['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
        print(f"📊 Max steps: {training_args.max_steps}")
        print(f"📊 Logging every: {training_args.logging_steps} steps")
        print(f"📊 Evaluation every: {training_args.eval_steps} steps")
        print("📊 Progress bars: ENABLED")
        print("📊 Loss reporting: ENABLED")
        print("=" * 60)
        
        # Train the model
        trainer.train()

        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        print(f"SUCCESS: Fine-tuning completed! Model saved to {self.output_dir}")

        # Save training info
        training_info = {
            "base_model": self.model_name,
            "training_examples": len(tokenized_dataset["train"]),
            "validation_examples": len(tokenized_dataset["test"]),
            "final_train_loss": trainer.state.log_history[-1].get("train_loss", 0.0),
            "best_eval_loss": min([log.get("eval_loss", float("inf")) for log in trainer.state.log_history if "eval_loss" in log], default=0.0),
            "total_training_steps": trainer.state.global_step,
            "training_completed": True
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

        print("SUCCESS: Fine-tuned model loaded successfully!")

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
                do_sample=False,  # Deterministic - override model's default do_sample=True
                temperature=None,  # Explicitly unset temperature (override model default)
                top_p=None,      # Explicitly unset top_p (override model default)
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

    def get_real_logprobs_confidence(self, text: str) -> Dict[str, Any]:
        """
        Get classification with proper logprobs-based confidence calculation.
        This method ensures consistent confidence calculation with the base model.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with prediction and confidence
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

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,  # Deterministic - override model's default do_sample=True
                temperature=None,  # Explicitly unset temperature (override model default)
                top_p=None,      # Explicitly unset top_p (override model default)
                pad_token_id=self.tokenizer.eos_token_id
            )

            if outputs.scores:
                logits = outputs.scores[0][0]
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Get token IDs for positive and negative
                pos_token_id = self.tokenizer.encode("positive", add_special_tokens=False)[0]
                neg_token_id = self.tokenizer.encode("negative", add_special_tokens=False)[0]

                # Get log probabilities
                pos_log_prob = log_probs[pos_token_id].item()
                neg_log_prob = log_probs[neg_token_id].item()

                # Convert to probabilities and normalize
                pos_prob = torch.exp(torch.tensor(pos_log_prob)).item()
                neg_prob = torch.exp(torch.tensor(neg_log_prob)).item()
                
                total_prob = pos_prob + neg_prob
                pos_prob_norm = pos_prob / total_prob
                neg_prob_norm = neg_prob / total_prob

                # Determine prediction and confidence
                if pos_prob_norm > neg_prob_norm:
                    prediction = "positive"
                    confidence = pos_prob_norm
                else:
                    prediction = "negative"
                    confidence = neg_prob_norm

                return {
                    'prediction': prediction,
                    'confidence': confidence
                }
            else:
                # Fallback if no scores available
                return {
                    'prediction': 'positive',
                    'confidence': 0.5
                }


def main():
    """Run the fine-tuning process."""
    print("FINE-TUNING: Fine-Tuning Llama 3.1 for Sentiment Classification")
    print("=" * 60)
    
    # Check for required dependencies
    try:
        import peft
        import datasets
        import accelerate
        # Only check bitsandbytes on CUDA systems
        if torch.cuda.is_available():
            import bitsandbytes
        elif torch.backends.mps.is_available():
            print("INFO: Using MPS (Apple Silicon) - bitsandbytes optimizations disabled")
        else:
            print("INFO: Using CPU - training will be slower")
    except ImportError as e:
        print(f"ERROR: Missing required dependency: {e}")
        print("Please install fine-tuning dependencies:")
        if torch.backends.mps.is_available():
            print("pip install peft datasets accelerate")
        else:
            print("pip install peft bitsandbytes datasets accelerate")
        return

    # Check available compute devices
    print("Checking Checking available compute devices...")
    if torch.backends.mps.is_available():
        print("SUCCESS: Apple Silicon GPU (MPS) available")
    if torch.cuda.is_available():
        print(f"SUCCESS: NVIDIA GPU available: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        print("WARNING:  No GPU available - training will be slow on CPU")

    # Initialize fine-tuner
    fine_tuner = SentimentFineTuner()

    # Run fine-tuning
    try:
        trainer = fine_tuner.fine_tune()

        print("\nCOMPLETED: Fine-tuning completed successfully!")
        print(f"Model saved to: {fine_tuner.output_dir}")

        # Test the fine-tuned model
        print("\nTesting Testing fine-tuned model...")
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
        print(f"ERROR: Fine-tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()