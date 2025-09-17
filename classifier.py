"""
Sentiment classifier using Ollama for local LLM inference.
"""
import requests
import json
from typing import List, Optional, Dict, Any
import time


class OllamaSentimentClassifier:
    """
    A sentiment classifier that uses Ollama to run local LLMs.
    """

    def __init__(self, model_name: str = "llama3.2", base_url: str = "http://localhost:11434"):
        """
        Initialize the classifier.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def _create_prompt(self, text: str, few_shot_examples: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Create a prompt for sentiment classification.

        Args:
            text: Text to classify
            few_shot_examples: Optional list of example classifications

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            "You are a sentiment classifier. Classify the sentiment of the given text as either 'positive', 'negative', or 'neutral'.",
            "Respond with only one word: positive, negative, or neutral.",
            ""
        ]

        # Add few-shot examples if provided
        if few_shot_examples:
            prompt_parts.append("Examples:")
            for example in few_shot_examples:
                prompt_parts.append(f"Text: {example['text']}")
                prompt_parts.append(f"Sentiment: {example['sentiment']}")
                prompt_parts.append("")

        prompt_parts.extend([
            f"Text: {text}",
            "Sentiment:"
        ])

        return "\n".join(prompt_parts)

    def _call_ollama(self, prompt: str, temperature: float = 0.1) -> Optional[str]:
        """
        Make a request to Ollama API.

        Args:
            prompt: The prompt to send to the model
            temperature: Temperature for generation (lower = more deterministic)

        Returns:
            Generated response text or None if failed
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "max_tokens": 10  # We only need one word
                }
            }

            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "").strip().lower()

        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing Ollama response: {e}")
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

        # Extract the first word and normalize
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

        # If we can't parse it clearly, return None
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
        response = self._call_ollama(prompt)
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
        Test if Ollama is running and the model is available.

        Returns:
            True if connection successful, False otherwise
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
    classifier = OllamaSentimentClassifier()

    # Test connection
    print("Testing Ollama connection...")
    if not classifier.test_connection():
        print("❌ Failed to connect to Ollama. Make sure Ollama is running and the model is available.")
        print("Try running: ollama pull llama3.2")
        return

    print("✅ Connected to Ollama successfully!")

    # Test basic classification
    test_texts = [
        "I love this!",
        "This is terrible.",
        "It's okay, I guess.",
        "That's so skibidi",  # Gen Z slang - should be uncertain
    ]

    print("\nTesting basic classifications:")
    for text in test_texts:
        result = classifier.classify_single(text)
        print(f"'{text}' → {result}")


if __name__ == "__main__":
    main()