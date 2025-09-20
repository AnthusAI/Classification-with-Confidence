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
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Any

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
                 fine_tuned_path: str = None):
        """
        Initialize the transformer logprobs classifier.

        Args:
            model_name: Hugging Face model identifier
            fine_tuned_path: Path to fine-tuned model (None for base model)
        """
        self.model_name = model_name
        self.fine_tuned_path = fine_tuned_path
        self.model = None
        self.tokenizer = None
        self.is_fine_tuned = fine_tuned_path is not None

        # Map various token variations to binary classification
        self.positive_tokens = ["yes", "Yes", "YES", "y", "Y", "positive", "Positive", "POSITIVE", "pos", "Pos", "POS", "true", "True", "TRUE", "1"]
        self.negative_tokens = ["no", "No", "NO", "n", "N", "negative", "Negative", "NEGATIVE", "neg", "Neg", "NEG", "false", "False", "FALSE", "0"]

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
        """Create a binary yes/no prompt for positive sentiment detection."""
        if self.is_fine_tuned:
            # Use the same format as fine-tuning for consistency
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Classify the sentiment of this text as either 'positive' or 'negative':

"{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        else:
            # Original format for base model
            return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Is this text positive in sentiment? Answer with only "yes" or "no".

Text: "{text}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def get_real_logprobs_confidence(self, text: str) -> Dict[str, Any]:
        """
        Get REAL token probabilities from Llama 3.1 model.

        Args:
            text: Text to classify

        Returns:
            Dictionary with real logprobs and confidence scores
        """
        self._load_model()

        prompt = self.create_classification_prompt(text)

        # Tokenize the prompt (no padding needed for single input)
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move inputs to same device as model
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

                log_probs = F.log_softmax(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)

                generated_token_id = outputs.sequences[0][-1]
                generated_token = self.tokenizer.decode([generated_token_id]).strip()

                positive_prob = 0.0
                negative_prob = 0.0
                positive_logprob = float('-inf')
                negative_logprob = float('-inf')

                token_details = {}

                # Use different token sets based on model type
                if self.is_fine_tuned:
                    # Fine-tuned models output "positive"/"negative" directly
                    positive_tokens = ["positive", "Positive", "POSITIVE"]
                    negative_tokens = ["negative", "Negative", "NEGATIVE"]
                else:
                    # Base models use yes/no format
                    positive_tokens = self.positive_tokens
                    negative_tokens = self.negative_tokens

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
                                positive_logprob = max(positive_logprob, logprob)  # Take max logprob
                            elif token_text in negative_tokens:
                                negative_prob += prob
                                negative_logprob = max(negative_logprob, logprob)  # Take max logprob

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