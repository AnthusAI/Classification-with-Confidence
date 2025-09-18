#!/usr/bin/env python3
"""
Quick examples showing confidence differences between standard English and Gen Z slang.

This script demonstrates the core hypothesis with a focused set of examples.
"""
from classifier import LlamaSentimentClassifier
from consistency_confidence import ConfidenceScorer


def run_confidence_comparison():
    """
    Run a focused comparison showing confidence differences.
    """
    print("🔬 Confidence Comparison: Standard English vs Gen Z Slang")
    print("=" * 60)

    # Initialize components
    classifier = LlamaSentimentClassifier()
    scorer = ConfidenceScorer(classifier)

    # Test connection
    print("Loading model...")
    if not classifier.test_connection():
        print("❌ Model not available. Check setup instructions.")
        print("Try: huggingface-cli login")
        return

    print("✅ Model loaded successfully!")
    print()

    # Define test cases that should show clear confidence differences
    high_confidence_examples = [
        {"text": "This is absolutely amazing!", "expected": "positive"},
        {"text": "I hate this terrible thing!", "expected": "negative"},
        {"text": "It's okay, nothing special.", "expected": "positive", "category": "weakly_positive"},
    ]

    low_confidence_examples = [
        {"text": "That's so skibidi!", "expected": "positive"},
        {"text": "This is mid, not gonna lie.", "expected": "negative"},
        {"text": "You're literally sigma energy!", "expected": "positive"},
    ]

    print("🔥 HIGH CONFIDENCE EXAMPLES (Standard English)")
    print("-" * 50)
    high_confidences = []

    for example in high_confidence_examples:
        print(f"\nText: '{example['text']}'")
        result = scorer.classify_with_confidence(example['text'], num_samples=5)

        if result['prediction']:
            high_confidences.append(result['confidence'])
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Distribution: {result['response_distribution']}")

            # Check accuracy
            correct = "✅" if result['prediction'] == example['expected'] else "❌"
            print(f"  Expected: {example['expected']} {correct}")
        else:
            print("  ❌ Failed to classify")

    avg_high = sum(high_confidences) / len(high_confidences) if high_confidences else 0.0

    print(f"\n📊 Average confidence on standard English: {avg_high:.3f}")

    print("\n" + "=" * 60)
    print("🤔 LOW CONFIDENCE EXAMPLES (Gen Z Slang)")
    print("-" * 50)
    low_confidences = []

    for example in low_confidence_examples:
        print(f"\nText: '{example['text']}'")
        result = scorer.classify_with_confidence(example['text'], num_samples=5)

        if result['prediction']:
            low_confidences.append(result['confidence'])
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Distribution: {result['response_distribution']}")

            # Note: We might not know the "correct" answer for slang
            print(f"  Expected: {example['expected']} (our guess)")
        else:
            print("  ❌ Failed to classify")

    avg_low = sum(low_confidences) / len(low_confidences) if low_confidences else 0.0

    print(f"\n📊 Average confidence on Gen Z slang: {avg_low:.3f}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("📈 CONFIDENCE COMPARISON RESULTS")
    print("=" * 60)
    print(f"Standard English avg confidence: {avg_high:.3f}")
    print(f"Gen Z slang avg confidence:     {avg_low:.3f}")
    print(f"Confidence gap:                {avg_high - avg_low:.3f}")

    if avg_high > avg_low + 0.1:
        print("\n✅ SUCCESS: Model shows higher confidence on familiar language!")
        print("   This demonstrates that consistency-based confidence scoring")
        print("   can detect when models are uncertain about unfamiliar terms.")
    elif avg_high > avg_low:
        print("\n📈 PARTIAL SUCCESS: Small confidence gap detected")
        print("   The model shows some uncertainty on slang, but the gap is small.")
        print("   This could be due to:")
        print("   • Model already knows some Gen Z terms")
        print("   • Small sample size")
        print("   • Natural variability in responses")
    else:
        print("\n📊 INCONCLUSIVE: No clear confidence gap")
        print("   This could mean:")
        print("   • The model is already familiar with this Gen Z slang")
        print("   • The model is consistently uncertain about everything")
        print("   • Different slang terms might show clearer differences")

    print(f"\n💡 Try running with different examples or more samples for clearer results!")
    print(f"   python examples.py")


if __name__ == "__main__":
    run_confidence_comparison()