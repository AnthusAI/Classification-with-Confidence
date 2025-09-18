#!/usr/bin/env python3
"""
Method 3: Combined Confidence Scoring

This module demonstrates how to combine logprobs-based confidence with
consistency-based confidence for optimal uncertainty estimation.

Usage:
    python combined_confidence.py

Requirements:
    - logprobs_confidence.py
    - consistency_confidence.py
    - classifier.py
"""
import math
from typing import Dict, List, Optional, Any, Callable
from logprobs_confidence import TransformerLogprobsClassifier
from consistency_confidence import ConfidenceScorer
from classifier import LlamaSentimentClassifier


class CombinedConfidenceScorer:
    """
    Combines multiple confidence estimation methods for optimal results.
    """

    def __init__(self, classifier: LlamaSentimentClassifier,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the combined confidence scorer.

        Args:
            classifier: The base classifier to use
            openai_api_key: Optional OpenAI API key for logprobs
        """
        self.classifier = classifier
        self.consistency_scorer = ConfidenceScorer(classifier)
        self.logprobs_scorer = TransformerLogprobsClassifier()

    def get_combined_confidence(self, text: str,
                              num_consistency_samples: int = 5,
                              use_openai_logprobs: bool = False,
                              model: str = "gpt-4") -> Dict[str, Any]:
        """
        Combine logprobs and consistency for comprehensive confidence estimation.

        Args:
            text: Text to classify
            num_consistency_samples: Number of samples for consistency analysis
            use_openai_logprobs: Whether to use real OpenAI logprobs
            model: OpenAI model to use (if use_openai_logprobs=True)

        Returns:
            Combined confidence analysis
        """
        print(f"🔬 Combined Confidence Analysis: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print("-" * 60)

        print("📊 Method 1: Token Probability Analysis")
        if use_openai_logprobs:
            print("  Warning: OpenAI logprobs not implemented, using Transformer logprobs instead")

        logprobs_result = self.logprobs_scorer.get_real_logprobs_confidence(text)

        logprobs_conf = logprobs_result.get('confidence', 0.0)
        print(f"  Logprobs confidence: {logprobs_conf:.3f}")
        print(f"  Token distribution: {logprobs_result.get('sentiment_probabilities', {})}")

        print(f"\n🔄 Method 2: Response Consistency Analysis")
        consistency_result = self.consistency_scorer.classify_with_confidence(
            text, num_samples=num_consistency_samples
        )
        consistency_conf = consistency_result.get('confidence', 0.0)
        print(f"  Consistency confidence: {consistency_conf:.3f}")
        print(f"  Response distribution: {consistency_result.get('response_distribution', {})}")

        # Method 3: Combine the methods
        print(f"\n⚖️  Method 3: Combined Scoring")
        combined_result = self._combine_confidence_scores(
            logprobs_conf, consistency_conf, text,
            logprobs_result, consistency_result
        )

        print(f"  Combined confidence: {combined_result['combined_confidence']:.3f}")
        print(f"  Weighting strategy: {combined_result['weighting_strategy']}")
        print(f"  Final prediction: {combined_result['prediction']}")

        return combined_result

    def _combine_confidence_scores(self, logprobs_conf: float, consistency_conf: float,
                                 text: str, logprobs_result: Dict[str, Any],
                                 consistency_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine logprobs and consistency confidence using adaptive weighting.

        Args:
            logprobs_conf: Confidence from token probabilities
            consistency_conf: Confidence from response consistency
            text: Original text (for context-aware weighting)
            logprobs_result: Full logprobs analysis
            consistency_result: Full consistency analysis

        Returns:
            Combined confidence analysis
        """
        # Adaptive weighting strategy
        if logprobs_conf > 0.9 and consistency_conf < 0.7:
            # High logprobs but low consistency - possible overconfidence
            weight_logprobs = 0.3
            weight_consistency = 0.7
            strategy = "overconfidence_detection"
        elif logprobs_conf < 0.5 and consistency_conf < 0.5:
            # Both methods show uncertainty - amplify the uncertainty
            weight_logprobs = 0.5
            weight_consistency = 0.5
            strategy = "uncertainty_amplification"
        elif abs(logprobs_conf - consistency_conf) > 0.3:
            # Methods disagree significantly - trust the more conservative one
            if logprobs_conf < consistency_conf:
                weight_logprobs = 0.7
                weight_consistency = 0.3
            else:
                weight_logprobs = 0.3
                weight_consistency = 0.7
            strategy = "disagreement_resolution"
        else:
            # Methods agree reasonably - balanced weighting
            weight_logprobs = 0.6
            weight_consistency = 0.4
            strategy = "balanced_weighting"

        combined_conf = (weight_logprobs * logprobs_conf +
                        weight_consistency * consistency_conf)

        final_prediction = consistency_result.get('prediction',
                                                logprobs_result.get('prediction'))

        return {
            'combined_confidence': combined_conf,
            'logprobs_confidence': logprobs_conf,
            'consistency_confidence': consistency_conf,
            'prediction': final_prediction,
            'weighting_strategy': strategy,
            'weights': {
                'logprobs': weight_logprobs,
                'consistency': weight_consistency
            },
            'individual_results': {
                'logprobs': logprobs_result,
                'consistency': consistency_result
            },
            'method': 'combined_confidence'
        }

    def calibrate_confidence(self, raw_confidence: float,
                           validation_data: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Calibrate raw confidence scores to improve reliability.

        Args:
            raw_confidence: Raw combined confidence score
            validation_data: Optional validation data for calibration

        Returns:
            Calibrated confidence score
        """
        if validation_data is None:
            return self._sigmoid_calibration(raw_confidence)
        else:
            # Implement proper calibration with validation data
            return self._platt_scaling_calibration(raw_confidence, validation_data)

    def _sigmoid_calibration(self, raw_conf: float) -> float:
        """
        Simple sigmoid-based calibration for demonstration.

        Args:
            raw_conf: Raw confidence score

        Returns:
            Calibrated confidence score
        """
        # Simple calibration: adjust for overconfidence
        adjusted = raw_conf * 0.8 + 0.1  # Reduce overconfidence
        return max(0.0, min(1.0, adjusted))

    def _platt_scaling_calibration(self, raw_conf: float,
                                 validation_data: List[Dict[str, Any]]) -> float:
        """
        Platt scaling calibration (placeholder implementation).

        Args:
            raw_conf: Raw confidence score
            validation_data: Validation examples with ground truth

        Returns:
            Calibrated confidence score
        """
        # Placeholder for proper Platt scaling implementation
        return self._sigmoid_calibration(raw_conf)


def demonstrate_combined_confidence():
    """
    Demonstrate combined confidence scoring with multiple examples.
    """
    print("🎯 Method 3: Combined Confidence Scoring Demonstration")
    print("=" * 70)

    classifier = LlamaSentimentClassifier()
    combined_scorer = CombinedConfidenceScorer(classifier)

    if not classifier.test_connection():
        print("❌ Cannot load model. Please ensure:")
        print("  1. Hugging Face access: huggingface-cli login")
        print("  2. Model access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        return

    print("✅ Model loaded successfully!")
    print()

    examples = [
        "This is absolutely amazing!",  # Expected: High confidence both methods
        "I am happy to be sad about this fantastic disaster",  # Expected: Low confidence
        "It's okay, nothing special",  # Expected: Low confidence (weakly positive)
        "That's so skibidi!"  # Expected: High confidence (Llama 3.1 knows this)
    ]

    for i, text in enumerate(examples, 1):
        print(f"📝 Example {i}/{len(examples)}")
        result = combined_scorer.get_combined_confidence(
            text,
            num_consistency_samples=3,  # Fewer samples for demo speed
            use_openai_logprobs=False
        )

        calibrated_conf = combined_scorer.calibrate_confidence(result['combined_confidence'])
        print(f"  📊 Calibrated confidence: {calibrated_conf:.3f}")
        print()

    print("💡 Key Insights from Combined Approach:")
    print("• Cross-validation: Use consistency to validate logprobs")
    print("• Overconfidence detection: High logprobs + low consistency = suspicious")
    print("• Uncertainty amplification: Both methods uncertain = very uncertain")
    print("• Adaptive weighting: Different strategies for different patterns")
    print("• Calibration: Convert raw scores to reliable probabilities")


def main():
    """
    Main demonstration of combined confidence scoring.
    """
    demonstrate_combined_confidence()


if __name__ == "__main__":
    main()