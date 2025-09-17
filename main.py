#!/usr/bin/env python3
"""
Main runner script for the Classification with Confidence demonstration.

This script orchestrates the full experiment showing how LLMs can provide
confidence scores through response consistency analysis.
"""
import sys
import time
from typing import Dict, Any
from classifier import OllamaSentimentClassifier
from confidence import ConfidenceScorer
from fine_tuning import FewShotLearning
from datasets import get_test_sets, get_few_shot_examples, print_dataset_summary


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'-' * 40}")
    print(f" {title}")
    print(f"{'-' * 40}")


def test_connection(classifier: OllamaSentimentClassifier) -> bool:
    """
    Test the connection to Ollama and provide helpful error messages.

    Args:
        classifier: The classifier to test

    Returns:
        True if connection successful, False otherwise
    """
    print_section("Testing Ollama Connection")

    print("Checking if Ollama is running and model is available...")

    if classifier.test_connection():
        print("✅ Successfully connected to Ollama!")
        print(f"   Model: {classifier.model_name}")
        print(f"   API URL: {classifier.api_url}")
        return True
    else:
        print("❌ Failed to connect to Ollama.")
        print("\nTroubleshooting steps:")
        print("1. Make sure Ollama is installed and running:")
        print("   curl -fsSL https://ollama.ai/install.sh | sh")
        print("   ollama serve")
        print()
        print("2. Make sure the model is available:")
        print(f"   ollama pull {classifier.model_name}")
        print()
        print("3. Check if Ollama is running on the correct port:")
        print("   ollama list")
        print("   curl http://localhost:11434/api/tags")
        return False


def demonstrate_basic_classification(classifier: OllamaSentimentClassifier):
    """
    Demonstrate basic classification without confidence scoring.

    Args:
        classifier: The classifier to use
    """
    print_section("Basic Classification Demo")

    test_examples = [
        "This is absolutely amazing!",
        "I hate this so much!",
        "It's okay, I guess.",
        "That's so skibidi",  # Gen Z slang
        "This is bussin fr!",  # Gen Z slang
    ]

    print("Testing basic sentiment classification:")
    print("(This shows what the model predicts, but not how confident it is)")
    print()

    for text in test_examples:
        print(f"Text: '{text}'")
        result = classifier.classify_with_retries(text, retries=2)
        print(f"  → {result or 'Failed to classify'}")
        print()


def demonstrate_confidence_scoring(scorer: ConfidenceScorer):
    """
    Demonstrate confidence scoring through consistency analysis.

    Args:
        scorer: The confidence scorer to use
    """
    print_section("Confidence Scoring Demo")

    print("Now we'll classify the same texts multiple times to measure confidence.")
    print("Higher consistency across multiple runs = higher confidence.")
    print()

    # Test a mix of obvious and unclear examples
    test_examples = [
        {"text": "This is absolutely amazing!", "expected_confidence": "HIGH"},
        {"text": "That's so skibidi", "expected_confidence": "LOW"},
        {"text": "I hate this terrible thing!", "expected_confidence": "HIGH"},
        {"text": "This is bussin fr!", "expected_confidence": "LOW"},
    ]

    for example in test_examples:
        text = example["text"]
        expected = example["expected_confidence"]

        print(f"Text: '{text}' (Expected confidence: {expected})")

        result = scorer.classify_with_confidence(text, num_samples=5)

        if result["prediction"]:
            confidence_level = "HIGH" if result["confidence"] >= 0.8 else "MEDIUM" if result["confidence"] >= 0.5 else "LOW"
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.2f} ({confidence_level})")
            print(f"  Response distribution: {result['response_distribution']}")

            # Check if our hypothesis was correct
            if expected == confidence_level:
                print("  ✅ Matches expected confidence level!")
            else:
                print(f"  ⚠️  Expected {expected} confidence, got {confidence_level}")
        else:
            print("  ❌ Failed to get prediction")

        print()


def demonstrate_few_shot_improvement(few_shot_learner: FewShotLearning):
    """
    Demonstrate how few-shot learning improves confidence on slang.

    Args:
        few_shot_learner: The few-shot learning system
    """
    print_section("Few-Shot Learning Improvement Demo")

    print("This demonstrates how providing examples of Gen Z slang can improve")
    print("the model's confidence when classifying similar unfamiliar terms.")
    print()

    # Focus on Gen Z examples that should benefit from few-shot learning
    from datasets import get_examples_by_category

    test_examples = [
        {"text": "That's so skibidi", "expected": "positive", "category": "gen_z_positive"},
        {"text": "This is mid, not gonna lie", "expected": "negative", "category": "gen_z_negative"},
        {"text": "You're literally sigma energy", "expected": "positive", "category": "gen_z_positive"},
    ]

    few_shot_examples = get_few_shot_examples()[:5]  # Use first 5 examples

    print("Few-shot examples we'll provide to the model:")
    for ex in few_shot_examples:
        print(f"  '{ex['text']}' → {ex['sentiment']}")
    print()

    # Test each example before and after few-shot learning
    for example in test_examples:
        text = example["text"]
        print(f"Testing: '{text}'")

        # Before few-shot learning
        before = few_shot_learner.confidence_scorer.classify_with_confidence(
            text, num_samples=3
        )

        # After few-shot learning
        after = few_shot_learner.confidence_scorer.classify_with_confidence(
            text, num_samples=3, few_shot_examples=few_shot_examples
        )

        if before["prediction"] and after["prediction"]:
            improvement = after["confidence"] - before["confidence"]
            print(f"  Before: {before['prediction']} (confidence: {before['confidence']:.2f})")
            print(f"  After:  {after['prediction']} (confidence: {after['confidence']:.2f})")
            print(f"  Improvement: {improvement:+.2f}")

            if improvement > 0.1:
                print("  ✅ Significant confidence improvement!")
            elif improvement > 0:
                print("  📈 Modest confidence improvement")
            else:
                print("  📉 No improvement (this can happen)")
        else:
            print("  ❌ Failed to get predictions")

        print()


def run_comprehensive_evaluation(scorer: ConfidenceScorer):
    """
    Run a comprehensive evaluation across all test categories.

    Args:
        scorer: The confidence scorer to use
    """
    print_section("Comprehensive Evaluation")

    test_sets = get_test_sets()

    print("Running systematic evaluation across all test categories...")
    print("This will take a few minutes as we test multiple samples per example.")
    print()

    # Test standard examples (should have high confidence)
    standard_examples = test_sets["standard"][:5]  # Limit for demo
    print("Testing STANDARD examples (expected: high confidence):")

    standard_results = scorer.batch_classify_with_confidence(
        [ex["text"] for ex in standard_examples],
        num_samples=3
    )

    avg_standard_confidence = sum(r["confidence"] for r in standard_results if r["prediction"]) / len([r for r in standard_results if r["prediction"]])

    print(f"Average confidence on standard examples: {avg_standard_confidence:.2f}")

    # Test Gen Z slang (should have lower confidence)
    slang_examples = test_sets["gen_z_slang"][:5]  # Limit for demo
    print(f"\nTesting GEN Z SLANG examples (expected: lower confidence):")

    slang_results = scorer.batch_classify_with_confidence(
        [ex["text"] for ex in slang_examples],
        num_samples=3
    )

    avg_slang_confidence = sum(r["confidence"] for r in slang_results if r["prediction"]) / len([r for r in slang_results if r["prediction"]])

    print(f"Average confidence on slang examples: {avg_slang_confidence:.2f}")

    # Compare results
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"Standard English confidence: {avg_standard_confidence:.2f}")
    print(f"Gen Z slang confidence: {avg_slang_confidence:.2f}")
    print(f"Confidence gap: {avg_standard_confidence - avg_slang_confidence:.2f}")

    if avg_standard_confidence > avg_slang_confidence + 0.1:
        print("✅ SUCCESS: Model shows higher confidence on familiar language!")
    else:
        print("⚠️  Results unclear - the confidence gap may be smaller than expected.")
        print("   This could happen due to:")
        print("   - Model already familiar with some Gen Z terms")
        print("   - Small sample size")
        print("   - Model being very consistent even when wrong")


def main():
    """
    Main function that orchestrates the full demonstration.
    """
    print_header("🔬 Classification with Confidence - Full Demonstration")

    print("This demonstration shows how to extract confidence scores from LLMs")
    print("by measuring response consistency across multiple queries.")
    print()
    print("We'll test the hypothesis that models show:")
    print("• HIGH confidence on familiar language (standard English)")
    print("• LOW confidence on unfamiliar language (Gen Z slang)")
    print("• IMPROVED confidence after few-shot learning")

    # Initialize components
    classifier = OllamaSentimentClassifier()
    scorer = ConfidenceScorer(classifier)
    few_shot_learner = FewShotLearning(classifier)

    # Test connection first
    if not test_connection(classifier):
        print("\n🛑 Cannot proceed without Ollama connection.")
        print("Please follow the troubleshooting steps above and try again.")
        sys.exit(1)

    # Show dataset information
    print_section("Dataset Overview")
    print_dataset_summary()

    # Run demonstrations in order
    try:
        demonstrate_basic_classification(classifier)
        demonstrate_confidence_scoring(scorer)
        demonstrate_few_shot_improvement(few_shot_learner)
        run_comprehensive_evaluation(scorer)

        print_header("🎉 Demonstration Complete!", "=")
        print("Key takeaways:")
        print("• LLMs can provide confidence scores through response consistency")
        print("• Models show lower confidence on unfamiliar terminology")
        print("• Few-shot learning can improve confidence on specific domains")
        print("• This approach works with any text generation model")
        print()
        print("Next steps you could try:")
        print("• Test with different Ollama models (ollama pull <model>)")
        print("• Add your own test examples to datasets.py")
        print("• Experiment with different confidence thresholds")
        print("• Try other classification tasks beyond sentiment")

    except KeyboardInterrupt:
        print("\n\n🛑 Demonstration interrupted by user.")
        print("You can restart anytime with: python main.py")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        print("This might be due to:")
        print("• Ollama connection issues")
        print("• Model not responding consistently")
        print("• Network timeout")
        print("\nTry running individual components to debug:")
        print("• python classifier.py")
        print("• python confidence.py")


if __name__ == "__main__":
    main()