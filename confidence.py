"""
Confidence scoring for LLM-based classification through response consistency analysis.
"""
import time
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import statistics
from classifier import OllamaSentimentClassifier


class ConfidenceScorer:
    """
    Measures classification confidence through response consistency.
    """

    def __init__(self, classifier: OllamaSentimentClassifier):
        """
        Initialize the confidence scorer.

        Args:
            classifier: The sentiment classifier to use
        """
        self.classifier = classifier

    def classify_with_confidence(self, text: str, num_samples: int = 5,
                               few_shot_examples: Optional[List[Dict[str, str]]] = None,
                               delay_between_calls: float = 0.1) -> Dict[str, Any]:
        """
        Classify text and compute confidence score based on consistency.

        Args:
            text: Text to classify
            num_samples: Number of independent classifications to run
            few_shot_examples: Optional examples for few-shot learning
            delay_between_calls: Delay between API calls to avoid overwhelming the model

        Returns:
            Dictionary containing:
                - prediction: Most common classification
                - confidence: Confidence score (0.0 to 1.0)
                - raw_responses: All individual responses
                - response_distribution: Count of each response
                - valid_responses: Number of successful classifications
        """
        raw_responses = []
        valid_responses = []

        print(f"Running {num_samples} classifications for: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        # Collect multiple independent classifications
        for i in range(num_samples):
            if i > 0 and delay_between_calls > 0:
                time.sleep(delay_between_calls)

            response = self.classifier.classify_single(text, few_shot_examples)
            raw_responses.append(response)

            if response is not None:
                valid_responses.append(response)

            # Show progress
            print(f"  Sample {i+1}/{num_samples}: {response}")

        # Handle case where no valid responses
        if not valid_responses:
            return {
                'prediction': None,
                'confidence': 0.0,
                'raw_responses': raw_responses,
                'response_distribution': {},
                'valid_responses': 0,
                'error': 'No valid responses received'
            }

        # Count response frequencies
        response_counts = Counter(valid_responses)
        most_common_response, most_common_count = response_counts.most_common(1)[0]

        # Calculate confidence as the proportion of responses that agree with the majority
        confidence = most_common_count / len(valid_responses)

        return {
            'prediction': most_common_response,
            'confidence': confidence,
            'raw_responses': raw_responses,
            'response_distribution': dict(response_counts),
            'valid_responses': len(valid_responses),
            'consensus_strength': most_common_count
        }

    def batch_classify_with_confidence(self, texts: List[str], num_samples: int = 5,
                                     few_shot_examples: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
        """
        Classify multiple texts with confidence scores.

        Args:
            texts: List of texts to classify
            num_samples: Number of samples per text
            few_shot_examples: Optional examples for few-shot learning

        Returns:
            List of classification results with confidence scores
        """
        results = []

        for i, text in enumerate(texts):
            print(f"\n--- Classifying {i+1}/{len(texts)} ---")
            result = self.classify_with_confidence(text, num_samples, few_shot_examples)
            result['text'] = text
            results.append(result)

        return results

    def evaluate_confidence_quality(self, test_examples: List[Dict[str, Any]],
                                  num_samples: int = 5,
                                  few_shot_examples: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Evaluate how well confidence scores correlate with accuracy.

        Args:
            test_examples: List of examples with 'text' and 'expected' keys
            num_samples: Number of samples per classification
            few_shot_examples: Optional examples for few-shot learning

        Returns:
            Evaluation metrics including accuracy by confidence level
        """
        results = []
        correct_predictions = 0
        total_predictions = 0

        for example in test_examples:
            text = example['text']
            expected = example['expected']

            result = self.classify_with_confidence(text, num_samples, few_shot_examples)

            if result['prediction'] is not None:
                is_correct = result['prediction'] == expected
                correct_predictions += int(is_correct)
                total_predictions += 1

                results.append({
                    'text': text,
                    'expected': expected,
                    'predicted': result['prediction'],
                    'confidence': result['confidence'],
                    'is_correct': is_correct,
                    'category': example.get('category', 'unknown')
                })

        # Calculate overall accuracy
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

        # Group by confidence levels
        confidence_buckets = {
            'high': [r for r in results if r['confidence'] >= 0.8],
            'medium': [r for r in results if 0.5 <= r['confidence'] < 0.8],
            'low': [r for r in results if r['confidence'] < 0.5]
        }

        # Calculate accuracy by confidence level
        bucket_stats = {}
        for bucket_name, bucket_results in confidence_buckets.items():
            if bucket_results:
                accuracy = sum(r['is_correct'] for r in bucket_results) / len(bucket_results)
                avg_confidence = statistics.mean(r['confidence'] for r in bucket_results)
                bucket_stats[bucket_name] = {
                    'count': len(bucket_results),
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence
                }
            else:
                bucket_stats[bucket_name] = {
                    'count': 0,
                    'accuracy': 0.0,
                    'avg_confidence': 0.0
                }

        return {
            'overall_accuracy': overall_accuracy,
            'total_examples': len(test_examples),
            'successful_predictions': total_predictions,
            'confidence_buckets': bucket_stats,
            'detailed_results': results
        }

    def compare_confidence_by_category(self, test_examples: List[Dict[str, Any]],
                                     num_samples: int = 5) -> Dict[str, Any]:
        """
        Compare confidence scores across different example categories.

        Args:
            test_examples: List of examples with 'text', 'expected', and 'category' keys
            num_samples: Number of samples per classification

        Returns:
            Statistics comparing confidence by category
        """
        category_results = {}

        # Group examples by category
        by_category = {}
        for example in test_examples:
            category = example.get('category', 'unknown')
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(example)

        # Evaluate each category
        for category, examples in by_category.items():
            print(f"\n=== Evaluating category: {category} ===")

            category_confidences = []
            category_accuracies = []

            for example in examples:
                result = self.classify_with_confidence(example['text'], num_samples)

                if result['prediction'] is not None:
                    category_confidences.append(result['confidence'])
                    is_correct = result['prediction'] == example['expected']
                    category_accuracies.append(is_correct)

            if category_confidences:
                category_results[category] = {
                    'count': len(examples),
                    'avg_confidence': statistics.mean(category_confidences),
                    'min_confidence': min(category_confidences),
                    'max_confidence': max(category_confidences),
                    'accuracy': statistics.mean(category_accuracies) if category_accuracies else 0.0,
                    'std_confidence': statistics.stdev(category_confidences) if len(category_confidences) > 1 else 0.0
                }
            else:
                category_results[category] = {
                    'count': len(examples),
                    'avg_confidence': 0.0,
                    'min_confidence': 0.0,
                    'max_confidence': 0.0,
                    'accuracy': 0.0,
                    'std_confidence': 0.0
                }

        return category_results


def main():
    """
    Test the confidence scoring functionality.
    """
    # Initialize classifier and confidence scorer
    classifier = OllamaSentimentClassifier()
    scorer = ConfidenceScorer(classifier)

    # Test connection
    if not classifier.test_connection():
        print("❌ Failed to connect to Ollama. Make sure it's running!")
        return

    print("✅ Testing confidence scoring...")

    # Test a few examples
    test_texts = [
        "This is absolutely amazing!",  # Should have high confidence
        "That's so skibidi",  # Should have low confidence
    ]

    for text in test_texts:
        print(f"\n--- Testing: '{text}' ---")
        result = scorer.classify_with_confidence(text, num_samples=3)

        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Response distribution: {result['response_distribution']}")


if __name__ == "__main__":
    main()