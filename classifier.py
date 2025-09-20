"""
Sentiment classifier using Hugging Face Transformers with Llama 3.1.
"""
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Dict, Any
import time
import warnings
warnings.filterwarnings("ignore")


class LlamaSentimentClassifier:
    """
    A sentiment classifier that uses Hugging Face Transformers with Llama 3.1.
    """

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the classifier.

        Args:
            model_name: Hugging Face model identifier for Llama 3.1
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the Llama model and tokenizer (lazy loading)."""
        if self.model is None:
            print(f"Loading {self.model_name}...")
            print("This may take a few minutes and requires significant memory...")

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16
                )

                if torch.backends.mps.is_available():
                    device = "mps"
                elif torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

                self.model = self.model.to(device)

                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                device = next(self.model.parameters()).device
                print(f"✅ Model loaded successfully on {device}!")

            except Exception as e:
                print(f"❌ Failed to load model: {e}")
                print("Make sure you have:")
                print("  1. Hugging Face token: huggingface-cli login")
                print("  2. Model access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
                print("  3. Required packages: pip install torch transformers accelerate")
                raise

    def _create_prompt(self, text: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Create a prompt for sentiment classification.

        Args:
            text: Text to classify
            few_shot_examples: Optional list of example classifications

        Returns:
            Formatted prompt string using Llama 3.1 chat format
        """
        # Build the system message
        system_msg = "You are a sentiment classifier. Classify the sentiment of the given text as either 'positive', 'negative', or 'neutral'. Respond with only one word."

        if few_shot_examples:
            system_msg += "\n\nExamples:"
            for example in few_shot_examples:
                system_msg += f"\nText: {example['text']}\nSentiment: {example['sentiment']}"

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

Text: {text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return prompt

    def _generate_response(self, prompt: str, temperature: float = 0.0) -> Optional[str]:
        """
        Generate response using the Hugging Face model.

        Args:
            prompt: The prompt to send to the model
            temperature: Temperature for generation (lower = more deterministic)

        Returns:
            Generated response text or None if failed
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Move inputs to same device as model
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                if temperature > 0:
                    # Sampling mode with temperature
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=True,
                        temperature=temperature,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                else:
                    # Deterministic mode (no temperature parameters)
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,  # Deterministic - no temperature needed
                        pad_token_id=self.tokenizer.eos_token_id
                    )

            # Decode response
            response = self.tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            return response.strip().lower()

        except Exception as e:
            print(f"Error generating response: {e}")
            return None

    def _normalize_response(self, response: str) -> Optional[str]:
        """
        Normalize and validate the model response.

        Args:
            response: Raw response from the model

        Returns:
            Normalized sentiment ('positive', 'negative', 'neutral') or None if invalid
        """
        if not response:
            return None

        first_word = response.split()[0].lower().strip('.,!?')

        # Map various response formats to standard labels
        positive_words = ['positive', 'pos', 'good', 'happy', '1']
        negative_words = ['negative', 'neg', 'bad', 'sad', '0']
        neutral_words = ['neutral', 'neut', 'mixed', 'unclear', '2']

        if first_word in positive_words:
            return 'positive'
        elif first_word in negative_words:
            return 'negative'
        elif first_word in neutral_words:
            return 'neutral'

        return None

    def classify_single(self, text: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """
        Classify a single text for sentiment.

        Args:
            text: Text to classify
            few_shot_examples: Optional examples for few-shot learning

        Returns:
            Sentiment classification ('positive', 'negative', 'neutral') or None if failed
        """
        prompt = self._create_prompt(text, few_shot_examples)
        response = self._generate_response(prompt)
        return self._normalize_response(response)

    def classify_with_retries(self, text: str, retries: int = 3,
                            few_shot_examples: Optional[List[Dict[str, str]]] = None) -> Optional[str]:
        """
        Classify text with retries for robustness.

        Args:
            text: Text to classify
            retries: Number of retry attempts
            few_shot_examples: Optional examples for few-shot learning

        Returns:
            Sentiment classification or None if all attempts failed
        """
        for attempt in range(retries):
            result = self.classify_single(text, few_shot_examples)
            if result is not None:
                return result

            if attempt < retries - 1:
                time.sleep(0.5)  # Brief delay between retries

        return None

    def test_connection(self) -> bool:
        """
        Test if the model is loaded and working.

        Returns:
            True if model works, False otherwise
        """
        try:
            # Try a simple classification
            result = self.classify_single("This is a test.")
            return result is not None
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


def main():
    """
    Simple test of the classifier.
    """
    classifier = LlamaSentimentClassifier()

    print("Testing model...")
    if not classifier.test_connection():
        print("❌ Failed to load model. Check setup instructions.")
        return

    print("✅ Model working successfully!")

    test_texts = [
        "I love this!",
        "This is terrible.",
        "It's okay, I guess.",
        "Best worst thing ever",  # Contradictory - should be uncertain
    ]

    print("\nTesting basic classifications:")
    for text in test_texts:
        result = classifier.classify_single(text)
        print(f"'{text}' → {result}")


if __name__ == "__main__":
    main()