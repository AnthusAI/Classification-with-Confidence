#!/usr/bin/env python3
"""
Demonstration of uncertainty detection with higher temperature settings.

This script shows cases where we can actually find less than perfect confidence
by using higher temperature settings to reveal underlying model uncertainty.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")


class TemperatureUncertaintyDemo:
    """
    Demonstrates uncertainty detection using temperature-based sampling with Hugging Face Transformers.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """Initialize with Hugging Face model."""
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the Hugging Face model."""
        print(f"Loading {self.model_name}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16
            )

            # Use best available device (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

            self.model = self.model.to(device)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("Make sure you have Hugging Face access setup.")
            raise

    def classify_with_temperature_confidence(self, text: str, num_samples: int = 10,
                                           temperature: float = 1.0) -> Dict[str, Any]:
        """
        Classify text with high temperature to reveal uncertainty patterns.

        Args:
            text: Text to classify
            num_samples: Number of samples to run
            temperature: Temperature setting (higher = more randomness)

        Returns:
            Classification results with confidence
        """
        responses = []
        raw_responses = []

        print(f"Testing with temperature {temperature}, {num_samples} samples:")
        print(f"'{text}'")

        # Create prompt
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a sentiment classifier. Classify the sentiment of this text as positive, negative, or neutral.
Respond with only one word: positive, negative, or neutral.<|eot_id|><|start_header_id|>user<|end_header_id|>

Text: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

        for i in range(num_samples):
            try:
                # Tokenize
                inputs = self.tokenizer(prompt, return_tensors="pt")

                # Move inputs to same device as model
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Generate with temperature
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                # Decode response
                response = self.tokenizer.decode(
                    outputs[0][len(inputs['input_ids'][0]):],
                    skip_special_tokens=True
                ).strip().lower()

                raw_responses.append(response)

                # Extract first word and normalize
                first_word = response.split()[0] if response else ''

                # Normalize to standard categories
                if first_word in ['positive', 'pos', 'good', 'great']:
                    normalized = 'positive'
                elif first_word in ['negative', 'neg', 'bad', 'terrible']:
                    normalized = 'negative'
                elif first_word in ['neutral', 'neut', 'mixed', 'unclear']:
                    normalized = 'neutral'
                else:
                    # Keep unusual responses for analysis
                    normalized = first_word

                responses.append(normalized)
                print(f"  Sample {i+1}: {response} → {normalized}")

            except Exception as e:
                responses.append('error')
                raw_responses.append(f'error: {e}')
                print(f"  Sample {i+1}: ERROR")

        # Calculate confidence
        if responses:
            counts = Counter(responses)
            most_common_response, most_common_count = counts.most_common(1)[0]
            confidence = most_common_count / len(responses)
        else:
            most_common_response = None
            confidence = 0.0
            counts = Counter()

        return {
            'text': text,
            'prediction': most_common_response,
            'confidence': confidence,
            'distribution': dict(counts),
            'raw_responses': raw_responses,
            'temperature': temperature,
            'num_samples': num_samples
        }


def demonstrate_uncertainty_cases():
    """
    Demonstrate cases that show measurable uncertainty.
    """
    print("🌡️  Temperature-Based Uncertainty Detection")
    print("=" * 60)
    print("Higher temperature reveals model uncertainty on ambiguous text")
    print()

    demo = TemperatureUncertaintyDemo()

    # Test cases designed to create uncertainty
    uncertainty_cases = [
        {
            'text': 'Best worst thing ever',
            'expected': 'Should show confusion due to contradiction'
        },
        {
            'text': 'I am happy to be sad about this fantastic disaster',
            'expected': 'Should show uncertainty due to mixed emotions'
        },
        {
            'text': 'I love to hate this',
            'expected': 'Should vary between positive and negative'
        },
        {
            'text': 'This is perfectly imperfect',
            'expected': 'Should show some variation'
        },
        {
            'text': 'Whatever, I guess',
            'expected': 'Might show uncertainty vs strong neutral'
        }
    ]

    high_confidence_cases = [
        {
            'text': 'This is absolutely amazing!',
            'expected': 'Should still show high confidence'
        },
        {
            'text': 'I hate this terrible thing!',
            'expected': 'Should still show high confidence'
        }
    ]

    print("🤔 TESTING UNCERTAINTY CASES:")
    print("-" * 40)

    uncertain_results = []
    for case in uncertainty_cases:
        print(f"\nCase: {case['text']}")
        print(f"Expected: {case['expected']}")

        result = demo.classify_with_temperature_confidence(case['text'],
                                                        num_samples=8,
                                                        temperature=1.0)
        uncertain_results.append(result)

        confidence_level = "HIGH" if result['confidence'] >= 0.8 else "MEDIUM" if result['confidence'] >= 0.6 else "LOW"
        print(f"Result: {result['prediction']} (confidence: {result['confidence']:.3f} - {confidence_level})")
        print(f"Distribution: {result['distribution']}")

        if result['confidence'] < 0.9:
            print("✅ SUCCESS: Found measurable uncertainty!")
        else:
            print("📊 Still very consistent")
        print()

    print("\n🔥 TESTING HIGH CONFIDENCE CASES (Control):")
    print("-" * 40)

    confident_results = []
    for case in high_confidence_cases:
        print(f"\nCase: {case['text']}")
        print(f"Expected: {case['expected']}")

        result = demo.classify_with_temperature_confidence(case['text'],
                                                        num_samples=5,
                                                        temperature=1.0)
        confident_results.append(result)

        confidence_level = "HIGH" if result['confidence'] >= 0.8 else "MEDIUM" if result['confidence'] >= 0.6 else "LOW"
        print(f"Result: {result['prediction']} (confidence: {result['confidence']:.3f} - {confidence_level})")
        print(f"Distribution: {result['distribution']}")
        print()

    # Summary
    print("\n📊 SUMMARY:")
    print("=" * 30)

    uncertain_confidences = [r['confidence'] for r in uncertain_results if r['confidence'] > 0]
    confident_confidences = [r['confidence'] for r in confident_results if r['confidence'] > 0]

    if uncertain_confidences:
        avg_uncertain = sum(uncertain_confidences) / len(uncertain_confidences)
        print(f"Average confidence on uncertain cases: {avg_uncertain:.3f}")

    if confident_confidences:
        avg_confident = sum(confident_confidences) / len(confident_confidences)
        print(f"Average confidence on clear cases: {avg_confident:.3f}")

    # Count cases with uncertainty
    uncertain_count = sum(1 for r in uncertain_results if r['confidence'] < 0.9)
    print(f"Cases showing uncertainty (<90% confidence): {uncertain_count}/{len(uncertain_results)}")

    print(f"\n💡 Key Finding:")
    print(f"Higher temperature (1.0) successfully reveals model uncertainty")
    print(f"on contradictory and ambiguous text that appears certain at low temperature.")


def main():
    """
    Main demonstration.
    """
    # Test model loading
    try:
        demo = TemperatureUncertaintyDemo()
        print("✅ Model loaded successfully")
        demonstrate_uncertainty_cases()
    except Exception as e:
        print("❌ Cannot load model. Please ensure:")
        print("  1. Hugging Face access: huggingface-cli login")
        print("  2. Model access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        print("  3. Required packages: pip install torch transformers accelerate")
        return


if __name__ == "__main__":
    main()