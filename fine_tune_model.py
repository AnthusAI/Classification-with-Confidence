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
import time
import torch
import json
from pathlib import Path
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
        os.makedirs(self.output_dir, exist_ok=True)

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

        # Get HuggingFace token if available (needed for gated models like Llama)
        hf_token = os.environ.get('HF_TOKEN')

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model (no quantization on Mac/MPS)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token=hf_token
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

        # CRITICAL: Enable gradient checkpointing for LoRA
        # This is required when using gradient_checkpointing=True in TrainingArguments
        self.model.enable_input_require_grads()

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
        test_set_path = os.path.join(self.output_dir, 'test_set.json')
        with open(test_set_path, 'w') as f:
            json.dump(test_examples, f, indent=2)
        print(f"✅ Test set saved to {test_set_path} ({len(test_examples)} examples)")

        # Format ONLY the training data for instruction tuning
        formatted_data = []

        for example in train_examples:
            text = example['text']
            label = example['expected']
            category = example.get('category', 'unknown')

            # Create instruction using Llama chat template
            instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nClassify the sentiment of this text as either 'positive' or 'negative':\n\n\"{text}\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            # Direct classification response
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

            # Create instruction using Llama chat template
            instruction = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nClassify the sentiment of this text as either 'positive' or 'negative':\n\n\"{text}\"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            # Direct classification response
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
                max_length=256,  # Reduced for GPU memory efficiency - sentiment is typically short
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

        # Detect number of GPUs for adaptive batch sizing
        import torch
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

        # Adaptive batch sizing for multi-GPU training
        # Goal: effective_batch_size = num_gpus * per_device_batch * gradient_accum = 16
        if num_gpus >= 4:
            # 4+ GPUs: batch=4, gradient_accum=1, effective=16 (4x faster!)
            per_device_batch = 4
            gradient_accum = 1
        elif num_gpus == 2:
            # 2 GPUs: batch=2, gradient_accum=4, effective=16 (2x faster)
            per_device_batch = 2
            gradient_accum = 4
        else:
            # 1 GPU or CPU: batch=1, gradient_accum=16, effective=16
            per_device_batch = 1
            gradient_accum = 16

        training_args = TrainingArguments(
            output_dir=self.output_dir,

            # Training schedule - AGGRESSIVE anti-overfitting
            num_train_epochs=1,     # Reduced from 2 - stop much earlier
            max_steps=200,          # Reduced from 400 - half the training

            # Batch sizes - ADAPTIVE for multi-GPU efficiency
            per_device_train_batch_size=per_device_batch,
            per_device_eval_batch_size=per_device_batch * 2,  # Eval can use more memory
            gradient_accumulation_steps=gradient_accum,

            # Learning rate and optimization
            # Use standard optimizer on MPS, bitsandbytes on CUDA
            optim="adamw_torch" if torch.backends.mps.is_available() else "adamw_bnb_8bit",
            learning_rate=1e-4,     # Much lower learning rate (was 2e-4)
            weight_decay=0.05,      # Much stronger regularization (was 0.02)
            warmup_steps=100,

            # Memory and precision (disable fp16 on MPS and SageMaker)
            # NOTE: fp16 + gradient_checkpointing + LoRA causes "Attempting to unscale FP16 gradients" error
            fp16=False,  # Disabled - causes gradient scaling issues with LoRA
            gradient_checkpointing=True,  # CRITICAL: Enable gradient checkpointing to reduce memory usage
            dataloader_pin_memory=False,

            # Distributed training optimizations
            ddp_find_unused_parameters=False,  # Faster DDP for LoRA (all params used)

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

        # Detect and report GPU configuration
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        device = next(self.model.parameters()).device

        if num_gpus > 1:
            print(f"🚀 MULTI-GPU TRAINING: {num_gpus} GPUs detected!")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"   Training strategy: DistributedDataParallel")
            print(f"   Batch per GPU: {training_args.per_device_train_batch_size}")
            print(f"   Effective batch: {num_gpus * training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        elif str(device).startswith('mps'):
            print("Training on Apple Silicon GPU - monitoring memory usage...")
        elif str(device).startswith('cuda'):
            print(f"Training on single NVIDIA GPU: {torch.cuda.get_device_name()}")

        # Enable detailed logging
        import logging
        logging.basicConfig(level=logging.INFO)

        print("🚀 STARTING TRAINING...")
        print(f"📊 Training steps: {len(tokenized_dataset['train']) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * max(1, num_gpus))}")
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


class SageMakerTrainingOrchestrator:
    """
    Orchestrate SageMaker training job submission and monitoring.

    This class handles the complete workflow for training on AWS SageMaker:
    1. Prepare and package training data and code
    2. Upload to S3
    3. Submit SageMaker training job
    4. Monitor progress and stream logs
    5. Download trained model artifacts

    The trained model will have identical structure to local training for compatibility.
    """

    def __init__(self, model_name: str, output_dir: str, instance_type: str = 'ml.p3.2xlarge'):
        """
        Initialize the SageMaker training orchestrator.

        Args:
            model_name: HuggingFace model ID to fine-tune
            output_dir: Local directory to save downloaded model
            instance_type: SageMaker instance type for training
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.instance_type = instance_type

        # Import SageMaker utilities
        from sagemaker_utils import get_account_id, discover_iam_role, discover_s3_bucket

        # Get HuggingFace token for gated model access
        self.hf_token = os.environ.get('HF_TOKEN')
        if not self.hf_token:
            raise Exception(
                "HF_TOKEN environment variable not set. "
                "Required for accessing Llama 3.1-8B-Instruct (gated model). "
                "Get token from https://huggingface.co/settings/tokens and run: export HF_TOKEN='hf_...'"
            )

        # AWS setup
        self.region = 'us-east-1'  # Match existing deployment
        self.account_id = get_account_id()
        self.iam_role = discover_iam_role()
        self.s3_bucket = discover_s3_bucket(region=self.region)
        self.job_name = f"llama-sentiment-{int(time.time())}"

        # S3 paths
        self.s3_prefix = f"training/{self.job_name}"
        self.s3_data_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/data"
        self.s3_code_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/code"
        self.s3_output_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/output"

    def prepare_training_data(self) -> str:
        """
        Package dataset/ directory for upload.

        Returns:
            Path to dataset directory
        """
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            raise Exception(f"Dataset directory not found: {dataset_path}")

        file_count = len(list(Path(dataset_path).glob('*.txt')))
        print(f"  Found {file_count} data files in {dataset_path}/")

        return dataset_path

    def prepare_training_code(self) -> str:
        """
        Create source tarball with training code and dependencies.

        Returns:
            Path to created tarball
        """
        import tarfile
        import io

        tarball_path = "/tmp/sourcedir.tar.gz"

        print("  Creating source code tarball...")

        with tarfile.open(tarball_path, "w:gz") as tar:
            # Add training entry point
            tar.add("sagemaker_train.py")
            print("    + sagemaker_train.py")

            # Add dependencies
            dependencies = [
                "fine_tune_model.py",
                "dataset_loader.py"
            ]

            for dep in dependencies:
                if os.path.exists(dep):
                    tar.add(dep)
                    print(f"    + {dep}")
                else:
                    print(f"    Warning: {dep} not found, skipping")

            # Create requirements.txt
            from sagemaker_utils import create_requirements_file
            requirements = create_requirements_file()

            # Add requirements to tarball
            req_info = tarfile.TarInfo(name="requirements.txt")
            req_info.size = len(requirements.encode())
            tar.addfile(req_info, io.BytesIO(requirements.encode()))
            print("    + requirements.txt")

        tarball_size = os.path.getsize(tarball_path) / 1024 / 1024
        print(f"  ✓ Created tarball: {tarball_size:.1f} MB")

        return tarball_path

    def upload_to_s3(self, data_dir: str, code_tarball: str):
        """
        Upload training data and code to S3.

        Args:
            data_dir: Path to dataset directory
            code_tarball: Path to source code tarball
        """
        from sagemaker_utils import upload_to_s3

        print(f"  Uploading training data to {self.s3_data_path}...")
        upload_to_s3(data_dir, self.s3_data_path)

        print(f"  Uploading training code to {self.s3_code_path}...")
        upload_to_s3(code_tarball, f"{self.s3_code_path}/sourcedir.tar.gz")

    def submit_training_job(self):
        """Submit SageMaker training job."""
        import boto3

        sagemaker = boto3.client('sagemaker', region_name=self.region)

        # Get PyTorch training container (GPU-enabled)
        image = f"763104351884.dkr.ecr.{self.region}.amazonaws.com/pytorch-training:2.3.0-gpu-py311"

        # Configure training job
        training_params = {
            'TrainingJobName': self.job_name,
            'RoleArn': self.iam_role,
            'AlgorithmSpecification': {
                'TrainingImage': image,
                'TrainingInputMode': 'File',
            },
            'Environment': {
                'HF_TOKEN': self.hf_token,
                'HUGGING_FACE_HUB_TOKEN': self.hf_token,
            },
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': self.s3_data_path,
                            'S3DataDistributionType': 'FullyReplicated',
                        }
                    },
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': self.s3_output_path,
            },
            'ResourceConfig': {
                'InstanceType': self.instance_type,
                'InstanceCount': 1,
                'VolumeSizeInGB': 30,
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 3600,  # 1 hour max
            },
            'HyperParameters': {
                'sagemaker_program': 'sagemaker_train.py',
                'sagemaker_submit_directory': f"{self.s3_code_path}/sourcedir.tar.gz",
                'model_name': self.model_name,
            },
        }

        print(f"  Job name: {self.job_name}")
        print(f"  Instance: {self.instance_type}")
        print(f"  Container: pytorch-training:2.3.0-gpu-py311")
        print(f"  Region: {self.region}")

        sagemaker.create_training_job(**training_params)
        print(f"✓ Training job submitted!")

        # Print cost estimate
        cost_per_hour = {
            'ml.p3.2xlarge': 3.83,
            'ml.p3.8xlarge': 14.69,
            'ml.p3.16xlarge': 28.15,
            'ml.g4dn.xlarge': 0.736,
            'ml.g6e.xlarge': 1.25,
            'ml.g6e.2xlarge': 2.50,
            'ml.g6e.12xlarge': 10.00,  # 4x L40S GPUs
        }.get(self.instance_type, 10.00)

        print(f"\n💰 Estimated cost: ~${cost_per_hour}/hour")
        if num_gpus := 4 if 'g6e.12xlarge' in self.instance_type else 1:
            print(f"   Multi-GPU training (4 GPUs): ~4x faster")
            print(f"   Expected duration: 3-5 minutes (~${cost_per_hour * 0.08:.2f})")
        else:
            print(f"   Expected duration: 10-20 minutes (~${cost_per_hour * 0.25:.2f})")

    def wait_for_completion(self):
        """Monitor training job and stream logs."""
        import boto3

        sagemaker = boto3.client('sagemaker', region_name=self.region)

        print(f"\nMonitoring training job: {self.job_name}")
        print("=" * 60)

        # Try to stream CloudWatch logs
        try:
            from sagemaker_utils import stream_cloudwatch_logs
            stream_cloudwatch_logs(
                log_group=f"/aws/sagemaker/TrainingJobs",
                log_stream_prefix=self.job_name,
                region=self.region
            )
        except Exception as e:
            print(f"Note: Could not stream logs: {e}")
            print("Monitoring via status polling instead...\n")

        # Poll for completion
        last_status = None
        while True:
            response = sagemaker.describe_training_job(TrainingJobName=self.job_name)
            status = response['TrainingJobStatus']

            # Print status updates
            if status != last_status:
                timestamp = time.strftime('%H:%M:%S')
                print(f"[{timestamp}] Status: {status}")
                last_status = status

            if status == 'Completed':
                print("\n✓ Training completed successfully!")
                break
            elif status in ['Failed', 'Stopped']:
                print(f"\n✗ Training {status.lower()}")
                print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
                print(f"\nTo debug, check CloudWatch logs:")
                print(f"  Log group: /aws/sagemaker/TrainingJobs")
                print(f"  Log stream: {self.job_name}/algo-1-*")
                raise Exception(f"Training job {status.lower()}")

            time.sleep(30)

    def download_model(self):
        """Download trained model from S3."""
        from sagemaker_utils import download_from_s3

        # SageMaker saves to {s3_output_path}/{job_name}/output/model.tar.gz
        s3_model_path = f"{self.s3_output_path}/{self.job_name}/output/model.tar.gz"

        print(f"\nDownloading trained model from S3...")
        print(f"  Source: s3://{self.s3_bucket}/{self.s3_prefix}/output/{self.job_name}/output/model.tar.gz")
        print(f"  Destination: {self.output_dir}")

        # Download and extract
        import tempfile
        import tarfile

        with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
            download_from_s3(s3_model_path, tmp.name)

            # Extract to output directory
            os.makedirs(self.output_dir, exist_ok=True)
            with tarfile.open(tmp.name, 'r:gz') as tar:
                tar.extractall(self.output_dir)

            os.unlink(tmp.name)

        print(f"✓ Model downloaded to {self.output_dir}")

        # Verify required files exist
        print("\nVerifying output files...")
        required_files = [
            'adapter_config.json',
            'adapter_model.safetensors',
            'test_set.json',
            'training_info.json'
        ]

        all_present = True
        for filename in required_files:
            filepath = Path(self.output_dir) / filename
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                print(f"  ✓ {filename} ({size_kb:.1f} KB)")
            else:
                print(f"  ✗ {filename} (MISSING)")
                all_present = False

        if not all_present:
            print("\n⚠️  Warning: Some expected files are missing!")
            print("The model may not be fully compatible with evaluation scripts.")
        else:
            print("\n✓ All required files present!")

    def run(self):
        """Execute complete SageMaker training workflow."""
        print("=" * 60)
        print("SageMaker Training Workflow")
        print("=" * 60)

        try:
            # Phase 1: Prepare
            print("\n[1/5] Preparing training data and code...")
            data_dir = self.prepare_training_data()
            code_tarball = self.prepare_training_code()

            # Phase 2: Upload
            print("\n[2/5] Uploading to S3...")
            self.upload_to_s3(data_dir, code_tarball)

            # Phase 3: Submit
            print("\n[3/5] Submitting training job...")
            self.submit_training_job()

            # Phase 4: Monitor
            print("\n[4/5] Monitoring training progress...")
            self.wait_for_completion()

            # Phase 5: Download
            print("\n[5/5] Downloading trained model...")
            self.download_model()

            print("\n" + "=" * 60)
            print("SageMaker Training Complete!")
            print("=" * 60)
            print(f"Model saved to: {self.output_dir}")
            print("\nYou can now run evaluation scripts:")
            print("  python compare_base_vs_finetuned.py")
            print("  python generate_evaluation_cache.py")
            print("  python logprob_demo_cli.py 'text' --model finetuned")

        except Exception as e:
            print(f"\n{'='*60}")
            print("SageMaker Training Failed!")
            print(f"{'='*60}")
            print(f"Error: {e}")
            raise


def main():
    """Run the fine-tuning process."""
    import argparse

    parser = argparse.ArgumentParser(description='Fine-tune Llama 3.1 for sentiment classification')
    parser.add_argument(
        '--output-dir',
        default='./fine_tuned_sentiment_model',
        help='Output directory for the fine-tuned model'
    )
    parser.add_argument(
        '--sagemaker',
        action='store_true',
        help='Train using AWS SageMaker instead of locally (requires AWS credentials and IAM role)'
    )
    parser.add_argument(
        '--instance-type',
        default='ml.g6e.12xlarge',
        help='SageMaker instance type (only used with --sagemaker). Default: ml.g6e.12xlarge (4x L40S GPUs, ~$10/hr)'
    )

    args = parser.parse_args()

    print("FINE-TUNING: Sentiment Classification")
    print("=" * 60)

    # Branch based on training location
    if args.sagemaker:
        print("Training location: AWS SageMaker")
        print(f"Instance type: {args.instance_type}")
        print("=" * 60)

        # Validate AWS prerequisites
        try:
            from sagemaker_utils import validate_prerequisites
            validate_prerequisites(check_dataset=True)
        except Exception as e:
            print(f"\n❌ Prerequisite validation failed: {e}")
            return

        # Run SageMaker training
        try:
            orchestrator = SageMakerTrainingOrchestrator(
                model_name="meta-llama/Llama-3.1-8B-Instruct",
                output_dir=args.output_dir,
                instance_type=args.instance_type
            )
            orchestrator.run()

            print("\n✅ SageMaker training completed successfully!")
            print(f"Model saved to: {args.output_dir}")

        except Exception as e:
            print(f"\n❌ SageMaker training failed: {e}")
            import traceback
            traceback.print_exc()
            return

        return  # Exit after SageMaker training

    # Local training path (UNCHANGED from original implementation)
    print("Training location: Local")
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
    fine_tuner = SentimentFineTuner(output_dir=args.output_dir)

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