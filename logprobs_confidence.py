#!/usr/bin/env python3
"""
Method 1: Token Probability Analysis using Hugging Face Transformers

This module demonstrates how to extract real confidence scores from LLM token probabilities
using Hugging Face Transformers with Llama 3.1.

Usage:
    python logprobs_confidence.py

Requirements:
    - transformers, torch, accelerate packages
    - Access to Llama models (huggingface-cli login required)
    - GPU with sufficient memory (16GB+ recommended for 8B model)
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Any
from classification_config import ClassificationConfig, ClassificationMode, PromptTemplates, extract_classification_from_tokens, get_classification_confidence

# Suppress transformers warnings about generation parameters
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import warnings
warnings.filterwarnings("ignore")

# Suppress transformers generation warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"


class TransformerLogprobsClassifier:
    """
    Real token probability extraction using Hugging Face Transformers.
    Supports both base models and fine-tuned models.
    """

    def __init__(self,
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 fine_tuned_path: str = None,
                 config: Optional[ClassificationConfig] = None):
        """
        Initialize the transformer logprobs classifier.

        Args:
            model_name: Hugging Face model identifier
            fine_tuned_path: Path to fine-tuned model (None for base model)
            config: Classification configuration (optional)
        """
        self.model_name = model_name
        self.fine_tuned_path = fine_tuned_path
        self.model = None
        self.tokenizer = None
        self.is_fine_tuned = fine_tuned_path is not None

        # Use provided config or default to first-token mode
        if config is None:
            config = ClassificationConfig(mode=ClassificationMode.FIRST_TOKEN)
        self.config = config

        # Map various token variations to binary classification (backward compatibility)
        self.positive_tokens = self.config.positive_tokens or ["yes", "Yes", "YES", "y", "Y", "positive", "Positive", "POSITIVE", "pos", "Pos", "POS", "true", "True", "TRUE", "1"]
        self.negative_tokens = self.config.negative_tokens or ["no", "No", "NO", "n", "N", "negative", "Negative", "NEGATIVE", "neg", "Neg", "NEG", "false", "False", "FALSE", "0"]

    def _load_model(self):
        """Load the Llama model and tokenizer (lazy loading)."""
        if self.model is None:
            if self.is_fine_tuned:
                print(f"Loading fine-tuned model from {self.fine_tuned_path}...")
            else:
                print(f"Loading {self.model_name}...")
                print("Loading model for real logprobs extraction...")

            try:
                # Load tokenizer
                if self.is_fine_tuned:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.fine_tuned_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

                # Load model
                # Determine best device first
                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

                if self.is_fine_tuned:
                    # Load fine-tuned model - NO 8-bit quantization on MPS
                    from peft import PeftModel
                    base_model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        # load_in_8bit=True  # REMOVED - incompatible with MPS
                    )
                    self.model = PeftModel.from_pretrained(base_model, self.fine_tuned_path)
                    # Move fine-tuned model to device
                    self.model = self.model.to(device)
                else:
                    # Load base model
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16
                    )
                    # Move base model to device
                    self.model = self.model.to(device)

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                device = next(self.model.parameters()).device
                model_type = "fine-tuned" if self.is_fine_tuned else "base"
                print(f"✅ {model_type.title()} model loaded successfully on {device}!")

            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                import traceback
                print("🔍 Full traceback:")
                traceback.print_exc()
                if self.is_fine_tuned:
                    print("Make sure the fine-tuned model exists at the specified path.")
                    print("Run fine_tune_model.py first to create the fine-tuned model.")
                else:
                    print("Make sure you have:")
                    print("  1. Get HF token: https://huggingface.co/settings/tokens")
                    print("  2. Request Llama access: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct")
                    print("  3. Set token: export HF_TOKEN=your_token_here")
                    print("  4. Sufficient GPU memory (16GB+ recommended)")
                    print("  5. Required packages installed")
                print("\n🔄 Falling back to realistic simulation for demonstration...")
                return False

    def _load_fine_tuned_model(self):
        """Load fine-tuned model with error handling."""
        try:
            import os
            if not os.path.exists(self.fine_tuned_path):
                print(f"❌ Fine-tuned model not found at {self.fine_tuned_path}")
                return False

            return True  # Success, actual loading happens in _load_model

        except ImportError:
            print("❌ PEFT library not installed. Install with: pip install peft")
            return False

    def create_classification_prompt(self, text: str) -> str:
        """Create a classification prompt based on configuration."""
        self._load_model()  # Ensure tokenizer is loaded

        if self.config.mode == ClassificationMode.FIRST_TOKEN:
            return PromptTemplates.create_first_token_prompt(self.tokenizer, text)
        else:
            return PromptTemplates.create_last_token_prompt(self.tokenizer, text)

    def create_chain_of_thought_prompt(self, text: str) -> str:
        """Create a chain-of-thought prompt that asks for explanation followed by YES/NO."""
        self._load_model()  # Ensure tokenizer is loaded
        return PromptTemplates.create_last_token_prompt(self.tokenizer, text)

    def get_real_logprobs_confidence(self, text: str, raw_prompt: bool = False) -> Dict[str, Any]:
        """
        Get REAL token probabilities from Llama 3.1 model.

        Args:
            text: Text to classify or use as raw prompt
            raw_prompt: If True, use text as raw prompt instead of wrapping in classification question

        Returns:
            Dictionary with real logprobs and confidence scores
        """
        self._load_model()

        if raw_prompt:
            prompt = text
        else:
            prompt = self.create_classification_prompt(text)

        # Tokenize the prompt (no padding needed for single input)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move inputs to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Set generation parameters based on classification mode
        if self.config.mode == ClassificationMode.FIRST_TOKEN:
            # For direct classification, only need 1 token
            max_new_tokens = 1
        else:
            # For chain-of-thought, need enough tokens for explanation + answer
            max_new_tokens = 100

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,  # Deterministic - override model's default do_sample=True
                temperature=None,  # Explicitly unset temperature (override model default)
                top_p=None,      # Explicitly unset top_p (override model default)
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            if outputs.scores:
                # Get the full generated sequence
                input_length = inputs['input_ids'].shape[1]
                generated_sequence = outputs.sequences[0][input_length:]
                generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)

                if self.config.mode == ClassificationMode.FIRST_TOKEN:
                    # For direct classification, analyze the first token
                    logits = outputs.scores[0][0]  # First (and only) token's logits
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = F.softmax(logits, dim=-1)

                    # Get the actual generated token
                    generated_token_id = outputs.sequences[0][-1]
                    generated_token = self.tokenizer.decode([generated_token_id]).strip()

                    positive_prob = 0.0
                    negative_prob = 0.0
                    positive_logprob = float('-inf')
                    negative_logprob = float('-inf')

                    # Use appropriate token sets
                    if self.is_fine_tuned:
                        positive_tokens = ["positive", "Positive", "POSITIVE"]
                        negative_tokens = ["negative", "Negative", "NEGATIVE"]
                    else:
                        positive_tokens = self.positive_tokens
                        negative_tokens = self.negative_tokens

                    token_details = {}

                    for token_text in (positive_tokens + negative_tokens):
                        token_ids = self.tokenizer.encode(token_text, add_special_tokens=False)
                        if token_ids:
                            token_id = token_ids[0]
                            if token_id < len(probs):
                                prob = probs[token_id].item()
                                logprob = log_probs[token_id].item()
                                token_details[token_text] = {'prob': prob, 'logprob': logprob}

                                if token_text in positive_tokens:
                                    positive_prob += prob
                                    positive_logprob = max(positive_logprob, logprob)
                                elif token_text in negative_tokens:
                                    negative_prob += prob
                                    negative_logprob = max(negative_logprob, logprob)

                else:
                    # For chain-of-thought, find the final classification token and extract its confidence
                    # Use compute_transition_scores to get proper token probabilities
                    transition_scores = self.model.compute_transition_scores(
                        outputs.sequences,
                        outputs.scores,
                        normalize_logits=True
                    )

                    # Find the last token that matches a classification word
                    final_classification_confidence = 0.0
                    final_token_found = False

                    for i, (token_id, score) in enumerate(zip(generated_sequence, transition_scores[0])):
                        token = self.tokenizer.decode([token_id], skip_special_tokens=True).strip().upper()
                        if token in ['YES', 'NO', 'POSITIVE', 'NEGATIVE']:
                            # Found a classification token - use its confidence
                            final_classification_confidence = torch.exp(score).item()
                            final_token_found = True

                    # Set probabilities based on final classification
                    if final_token_found:
                        # Extract actual classification from generated text
                        classification = self.extract_final_classification([self.tokenizer.decode([token_id]) for token_id in generated_sequence])

                        if classification == 'positive':
                            positive_prob = final_classification_confidence
                            negative_prob = 1.0 - final_classification_confidence
                        else:
                            negative_prob = final_classification_confidence
                            positive_prob = 1.0 - final_classification_confidence
                    else:
                        # Fallback if no classification token found
                        positive_prob = 0.5
                        negative_prob = 0.5

                    positive_logprob = np.log(positive_prob) if positive_prob > 0 else float('-inf')
                    negative_logprob = np.log(negative_prob) if negative_prob > 0 else float('-inf')

                    token_details = {
                        'chain_of_thought': True,
                        'generated_text': generated_text,
                        'final_classification_confidence': final_classification_confidence if final_token_found else 0.5
                    }

                    # For consistency with the rest of the method, set up dummy variables
                    # since we don't need the full vocab analysis for chain-of-thought
                    probs = torch.zeros(self.tokenizer.vocab_size)  # Dummy for compatibility
                    log_probs = torch.zeros(self.tokenizer.vocab_size)  # Dummy for compatibility

                    # Set generated_token for chain-of-thought (use the final classification)
                    classification = self.extract_final_classification([self.tokenizer.decode([token_id]) for token_id in generated_sequence])
                    generated_token = "YES" if classification == 'positive' else "NO"

                sentiment_probs = {
                    'positive': positive_prob,
                    'negative': negative_prob
                }
                sentiment_logprobs = {
                    'positive': positive_logprob if positive_logprob != float('-inf') else -999.0,
                    'negative': negative_logprob if negative_logprob != float('-inf') else -999.0
                }

                top_k = 20
                top_probs, top_indices = torch.topk(probs, top_k)

                all_top_tokens = {}
                all_top_logprobs = {}

                for prob, idx in zip(top_probs, top_indices):
                    token = self.tokenizer.decode([idx.item()]).strip()
                    all_top_tokens[token] = prob.item()
                    all_top_logprobs[token] = log_probs[idx].item()

                if raw_prompt:
                    # For raw prompts, use all top tokens as token_details
                    token_details = {}
                    for prob, idx in zip(top_probs, top_indices):
                        token = self.tokenizer.decode([idx.item()]).strip()
                        token_details[token] = {
                            'prob': prob.item(),
                            'logprob': log_probs[idx].item()
                        }
                    
                    # For raw prompts, return the most likely token as prediction
                    most_likely_token = self.tokenizer.decode([top_indices[0].item()]).strip()
                    
                    return {
                        'text': text,
                        'prediction': most_likely_token,  # Most likely next token
                        'confidence': top_probs[0].item(),  # Probability of most likely token
                        'generated_token': generated_token,
                        'generated_classification': 'raw_prompt',
                        'token_details': token_details,
                        'all_top_tokens': all_top_tokens,
                        'all_top_logprobs': all_top_logprobs,
                        'method': 'real_transformer_logprobs_raw',
                        'model': self.model_name
                    }
                else:
                    # Original classification logic
                    predicted_sentiment = max(sentiment_probs, key=sentiment_probs.get)
                    confidence = sentiment_probs[predicted_sentiment]

                    # Also classify the generated token for comparison
                    generated_classification = self._classify_token(generated_token)

                    return {
                        'text': text,
                        'prediction': predicted_sentiment,
                        'confidence': confidence,
                        'generated_token': generated_token,
                        'generated_classification': generated_classification,
                        'sentiment_probabilities': sentiment_probs,
                        'sentiment_logprobs': sentiment_logprobs,
                        'token_details': token_details,
                        'all_top_tokens': all_top_tokens,
                        'all_top_logprobs': all_top_logprobs,
                        'method': 'real_transformer_logprobs',
                        'model': self.model_name
                    }

        return {'error': 'Failed to generate logprobs - no scores returned'}

    def _classify_token(self, token: str) -> str:
        """Classify a generated token into positive/negative/unknown."""
        token_clean = token.strip()

        if self.is_fine_tuned:
            # Fine-tuned models output "positive"/"negative" directly
            if token_clean.lower() in ['positive', 'pos']:
                return 'positive'
            elif token_clean.lower() in ['negative', 'neg']:
                return 'negative'
            else:
                return f'unknown_token:{token_clean}'
        else:
            # Base models use yes/no format
            if token_clean in self.positive_tokens:
                return 'positive'
            elif token_clean in self.negative_tokens:
                return 'negative'
            else:
                return f'unknown_token:{token_clean}'

    def extract_final_classification(self, generated_tokens: List[str]) -> str:
        """Extract final classification from generated tokens using config-based logic."""
        return extract_classification_from_tokens(generated_tokens, self.config)

    def get_multi_token_logprobs(self, text: str, max_new_tokens: int = None, raw_prompt: bool = False, chain_of_thought: bool = False) -> Dict[str, Any]:
        """
        Generate multiple tokens and return logprobs for each token generation step.

        Args:
            text: Input text to analyze
            max_new_tokens: Maximum number of tokens to generate
            raw_prompt: If True, use text as raw prompt; if False, wrap in classification prompt
            chain_of_thought: If True, use chain-of-thought prompt format

        Returns:
            Dict containing token-by-token logprob analysis
        """
        self._load_model()

        device = next(self.model.parameters()).device

        # Prepare prompt
        if raw_prompt:
            prompt = text
        elif chain_of_thought:
            prompt = self.create_chain_of_thought_prompt(text)
        else:
            prompt = self.create_classification_prompt(text)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]

            generation_params = {
                "return_dict_in_generate": True,
                "output_scores": True,
                "do_sample": False,
                "temperature": 0,  # Explicitly set temperature to 0
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "max_length": None  # Remove default max_length limit
            }

            # Only set max_new_tokens if specified, otherwise let model stop naturally
            if max_new_tokens is not None:
                generation_params["max_new_tokens"] = max_new_tokens
            else:
                # If no max_new_tokens specified, set a reasonable upper bound but let EOS stop it naturally
                generation_params["max_new_tokens"] = 100

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_params)

            if not outputs.scores:
                return {'error': 'No scores returned from model generation'}

            # Extract generated tokens
            generated_sequence = outputs.sequences[0][input_length:]  # Remove input tokens
            generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)

            # Analyze each token generation step
            token_analyses = []

            for step_idx, logits in enumerate(outputs.scores):
                if step_idx >= len(generated_sequence):
                    break

                # Get probabilities for this step
                step_logits = logits[0]  # Remove batch dimension
                log_probs = F.log_softmax(step_logits, dim=-1)
                probs = F.softmax(step_logits, dim=-1)

                # Get the actually generated token
                generated_token_id = generated_sequence[step_idx]
                generated_token = self.tokenizer.decode([generated_token_id]).strip()

                # Get top-k tokens for this step
                top_k = 12  # Show top 12 as requested
                top_probs, top_indices = torch.topk(probs, top_k)

                step_analysis = {
                    'step': step_idx + 1,
                    'generated_token': generated_token,
                    'generated_token_id': generated_token_id.item(),
                    'top_tokens': []
                }

                # Collect top tokens and their probabilities
                for rank, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                    token = self.tokenizer.decode([idx.item()]).strip()
                    step_analysis['top_tokens'].append({
                        'rank': rank + 1,
                        'token': token,
                        'token_id': idx.item(),
                        'probability': prob.item(),
                        'log_probability': log_probs[idx].item(),
                        'percentage': prob.item() * 100,
                        'is_generated': idx.item() == generated_token_id.item()
                    })

                token_analyses.append(step_analysis)

            return {
                'text': text,
                'prompt': prompt,
                'generated_text': generated_text,
                'generated_tokens': [self.tokenizer.decode([token_id]).strip() for token_id in generated_sequence],
                'token_analyses': token_analyses,
                'method': 'multi_token_logprobs',
                'model': self.model_name,
                'max_new_tokens': max_new_tokens,
                'raw_prompt': raw_prompt,
                'chain_of_thought': chain_of_thought
            }

        except Exception as e:
            return {'error': f'Failed to generate multi-token logprobs: {str(e)}'}

    def test_model_availability(self) -> bool:
        """Test if the model can be loaded."""
        try:
            self._load_model()
            return True
        except Exception as e:
            print(f"Model not available: {e}")
            return False


def demonstrate_real_logprobs():
    """
    Demonstrate REAL logprobs extraction from Llama 3.1.
    """
    print("🧠 Method 1: REAL Token Probability Analysis (Llama 3.1)")
    print("=" * 70)
    print("This shows ACTUAL token probabilities from Hugging Face Transformers")
    print()

    # Test both base and fine-tuned models if available
    import os
    fine_tuned_path = "./fine_tuned_sentiment_model"
    has_fine_tuned = os.path.exists(fine_tuned_path)

    if has_fine_tuned:
        print("🎯 Found fine-tuned model! Comparing base vs fine-tuned...")
        classifiers = {
            'Base Model': TransformerLogprobsClassifier(),
            'Fine-Tuned Model': TransformerLogprobsClassifier(fine_tuned_path=fine_tuned_path)
        }
    else:
        print("Using base model only (run fine_tune_model.py to create fine-tuned version)")
        classifiers = {
            'Base Model': TransformerLogprobsClassifier()
        }

    # Test model availability
    available_classifiers = {}
    for name, classifier in classifiers.items():
        print(f"🔍 Testing {name} availability...")
        if classifier.test_model_availability():
            available_classifiers[name] = classifier
        else:
            print(f"❌ {name} not available")

    if not available_classifiers:
        print("\n💡 To use real logprobs, you need:")
        print("  1. GPU with 16GB+ memory")
        print("  2. Hugging Face account: huggingface-cli login")
        print("  3. Access to Llama models")
        print("  4. Install: pip install torch transformers accelerate")
        print("\n🔄 Falling back to simulated examples for demonstration...")
        demonstrate_simulated_logprobs()
        return

    examples = [
        # HIGH CONFIDENCE EXAMPLES (>99%)
        {
            'text': "I absolutely love this amazing movie!",
            'description': "HIGH CONFIDENCE POSITIVE - Expect ~100% confidence"
        },
        {
            'text': "This film is terrible and boring.",
            'description': "HIGH CONFIDENCE NEGATIVE - Expect ~99% confidence"
        },

        # LOW CONFIDENCE EXAMPLES (52-77%)
        {
            'text': "Best worst thing ever",
            'description': "ULTRA-LOW CONFIDENCE - Expect ~52% confidence"
        },
        {
            'text': "Loving the hate",
            'description': "ULTRA-LOW CONFIDENCE - Expect ~52% confidence"
        },
        {
            'text': "This exists",
            'description': "LOW CONFIDENCE - Expect ~57% confidence"
        },
        {
            'text': "Sure okay",
            'description': "LOW CONFIDENCE - Expect ~64% confidence"
        },
        {
            'text': "Stuff exists",
            'description': "LOW CONFIDENCE - Expect ~68% confidence"
        },
        {
            'text': "Things happen",
            'description': "MEDIUM CONFIDENCE - Expect ~77% confidence"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"📊 REAL EXAMPLE {i}: {example['description']}")
        print(f"Text: \"{example['text']}\"")
        print("-" * 60)

        # Test each available model
        for model_name, classifier in available_classifiers.items():
            print(f"\n🤖 {model_name.upper()} RESULTS:")

            result = classifier.get_real_logprobs_confidence(example['text'])

            if 'error' not in result:
                print(f"🎯 Prediction: {result['prediction']} (confidence: {result['confidence']:.4f})")
                print(f"Generated token: '{result['generated_token']}' → {result['generated_classification']}")

                print(f"📋 Binary Classification Probabilities:")
                for sentiment, prob in result['sentiment_probabilities'].items():
                    logprob = result['sentiment_logprobs'][sentiment]
                    marker = " ← SELECTED" if sentiment == result['prediction'] else ""
                    print(f"  {sentiment:<10} {prob:>8.4f} (logprob: {logprob:>7.3f}){marker}")

                if result['token_details']:
                    print(f"🔍 Top Token Details:")
                    for token, details in list(result['token_details'].items())[:5]:
                        if classifier.is_fine_tuned:
                            class_type = "POS" if token.lower() in ['positive', 'pos'] else "NEG"
                        else:
                            class_type = "POS" if token in classifier.positive_tokens else "NEG"
                        print(f"  '{token:<8}' {details['prob']:>8.4f} (logprob: {details['logprob']:>7.3f}) [{class_type}]")

                # Analysis
                if result['confidence'] > 0.8:
                    confidence_level = "HIGH CONFIDENCE ✅"
                elif result['confidence'] > 0.5:
                    confidence_level = "MEDIUM CONFIDENCE ⚠️"
                else:
                    confidence_level = "LOW CONFIDENCE ❓"

                print(f"📈 {confidence_level}: {result['confidence']:.1%}")

            else:
                print(f"❌ Error: {result['error']}")

        print("=" * 70)
        print()

    print("💡 KEY INSIGHTS FROM REAL LOGPROBS:")
    print("• These are ACTUAL probabilities from Llama 3.1")
    print("• Logprobs are negative (natural log of probabilities)")
    print("• Higher probability = less negative logprob")
    print("• Model uncertainty shows as distributed probabilities")
    print("• Use these exact numbers for README examples!")


def demonstrate_simulated_logprobs():
    """
    Fallback demonstration with realistic simulated logprobs.
    """
    print("🔄 SIMULATED Token Probability Analysis")
    print("=" * 70)
    print("(Realistic simulation - shows what real logprobs would look like)")
    print()

    examples = [
        {
            'text': "I absolutely love this amazing movie!",
            'probs': {"positive": 0.923, "neutral": 0.051, "negative": 0.026},
            'description': "HIGH CONFIDENCE - Clear positive"
        },
        {
            'text': "This film is terrible and boring.",
            'probs': {"negative": 0.887, "neutral": 0.089, "positive": 0.024},
            'description': "HIGH CONFIDENCE - Clear negative"
        },
        {
            'text': "I am happy to be sad about this fantastic disaster",
            'probs': {"positive": 0.342, "negative": 0.331, "neutral": 0.327},
            'description': "LOW CONFIDENCE - Contradictory text"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"📊 SIMULATED EXAMPLE {i}: {example['description']}")
        print(f"Text: \"{example['text']}\"")
        print("-" * 50)

        prediction = max(example['probs'], key=example['probs'].get)
        confidence = example['probs'][prediction]

        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.3f}")
        print()

        print("Token Probabilities:")
        for sentiment, prob in example['probs'].items():
            logprob = torch.log(torch.tensor(prob)).item()
            marker = " ← SELECTED" if sentiment == prediction else ""
            print(f"  {sentiment:<10} {prob:>6.3f} (logprob: {logprob:>6.3f}){marker}")

        print("=" * 50)
        print()


def main():
    """
    Main demonstration of real logprobs extraction.
    """
    demonstrate_real_logprobs()


if __name__ == "__main__":
    main()