"""
Test datasets for sentiment classification with confidence scoring.
"""
from typing import List, Dict, Any


# Standard English examples - should have HIGH confidence
STANDARD_EXAMPLES = [
    # Clear positive sentiment
    {"text": "This is absolutely amazing!", "expected": "positive", "category": "obvious_positive"},
    {"text": "I love this so much!", "expected": "positive", "category": "obvious_positive"},
    {"text": "This is fantastic and wonderful!", "expected": "positive", "category": "obvious_positive"},
    {"text": "Perfect! Exactly what I wanted.", "expected": "positive", "category": "obvious_positive"},
    {"text": "Outstanding work, really impressive!", "expected": "positive", "category": "obvious_positive"},

    # Clear negative sentiment
    {"text": "This is absolutely terrible!", "expected": "negative", "category": "obvious_negative"},
    {"text": "I hate this so much!", "expected": "negative", "category": "obvious_negative"},
    {"text": "This is horrible and awful!", "expected": "negative", "category": "obvious_negative"},
    {"text": "Worst thing ever, complete disaster.", "expected": "negative", "category": "obvious_negative"},
    {"text": "Disgusting and unacceptable!", "expected": "negative", "category": "obvious_negative"},

    # Clear neutral sentiment
    {"text": "It's okay, I guess.", "expected": "neutral", "category": "obvious_neutral"},
    {"text": "This is fine, nothing special.", "expected": "neutral", "category": "obvious_neutral"},
    {"text": "Average quality, meets expectations.", "expected": "neutral", "category": "obvious_neutral"},
    {"text": "Neither good nor bad.", "expected": "neutral", "category": "obvious_neutral"},
    {"text": "Acceptable but unremarkable.", "expected": "neutral", "category": "obvious_neutral"},
]


# Gen Z / Gen Alpha slang - should have LOW confidence
GEN_Z_SLANG_EXAMPLES = [
    # Positive slang (but models might not know)
    {"text": "That's so skibidi!", "expected": "positive", "category": "gen_z_positive"},
    {"text": "This is absolutely bussin fr!", "expected": "positive", "category": "gen_z_positive"},
    {"text": "You're literally sigma energy!", "expected": "positive", "category": "gen_z_positive"},
    {"text": "That's fire and no cap!", "expected": "positive", "category": "gen_z_positive"},
    {"text": "This hits different, it's giving main character!", "expected": "positive", "category": "gen_z_positive"},
    {"text": "Ohio? More like this is rizz!", "expected": "positive", "category": "gen_z_positive"},
    {"text": "That's so slay, periodt!", "expected": "positive", "category": "gen_z_positive"},

    # Negative slang
    {"text": "This is mid, not gonna lie.", "expected": "negative", "category": "gen_z_negative"},
    {"text": "That's sus af and cringe.", "expected": "negative", "category": "gen_z_negative"},
    {"text": "Bruh this is straight trash, L take.", "expected": "negative", "category": "gen_z_negative"},
    {"text": "This ain't it chief, major ick.", "expected": "negative", "category": "gen_z_negative"},
    {"text": "Yikes, this is giving Ohio energy.", "expected": "negative", "category": "gen_z_negative"},
    {"text": "That's cap and lowkey toxic.", "expected": "negative", "category": "gen_z_negative"},

    # Neutral/confusing slang
    {"text": "It's giving vibes but make it Ohio.", "expected": "neutral", "category": "gen_z_neutral"},
    {"text": "That's lowkey mid but highkey sus.", "expected": "neutral", "category": "gen_z_neutral"},
    {"text": "This is literally just existing, bestie.", "expected": "neutral", "category": "gen_z_neutral"},
    {"text": "It's giving nothing, just vibes.", "expected": "neutral", "category": "gen_z_neutral"},
]


# Few-shot learning examples for improving Gen Z understanding
FEW_SHOT_EXAMPLES = [
    {"text": "That's bussin!", "sentiment": "positive"},
    {"text": "This is so skibidi", "sentiment": "positive"},
    {"text": "You have sigma energy", "sentiment": "positive"},
    {"text": "That's fire fr", "sentiment": "positive"},
    {"text": "This is mid", "sentiment": "negative"},
    {"text": "That's sus and cringe", "sentiment": "negative"},
    {"text": "Major L energy", "sentiment": "negative"},
    {"text": "This ain't it", "sentiment": "negative"},
]


def get_test_sets() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get organized test sets for confidence evaluation.

    Returns:
        Dictionary with different test categories
    """
    return {
        "standard": STANDARD_EXAMPLES,
        "gen_z_slang": GEN_Z_SLANG_EXAMPLES,
        "all": STANDARD_EXAMPLES + GEN_Z_SLANG_EXAMPLES
    }


def get_examples_by_category(category: str) -> List[Dict[str, Any]]:
    """
    Get examples filtered by category.

    Args:
        category: Category to filter by (e.g., 'obvious_positive', 'gen_z_positive')

    Returns:
        List of examples matching the category
    """
    all_examples = STANDARD_EXAMPLES + GEN_Z_SLANG_EXAMPLES
    return [ex for ex in all_examples if ex.get("category") == category]


def get_few_shot_examples() -> List[Dict[str, str]]:
    """
    Get few-shot learning examples for improving Gen Z slang understanding.

    Returns:
        List of example text-sentiment pairs
    """
    return FEW_SHOT_EXAMPLES


def print_dataset_summary():
    """
    Print a summary of the available datasets.
    """
    test_sets = get_test_sets()

    print("Dataset Summary:")
    print("=" * 50)

    for name, examples in test_sets.items():
        if name == "all":
            continue

        print(f"\n{name.upper()} ({len(examples)} examples):")

        # Count by expected sentiment
        sentiment_counts = {}
        category_counts = {}

        for ex in examples:
            sent = ex["expected"]
            cat = ex["category"]
            sentiment_counts[sent] = sentiment_counts.get(sent, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print(f"  Sentiment distribution: {sentiment_counts}")
        print(f"  Category distribution: {category_counts}")

        # Show a few examples
        print("  Sample examples:")
        for ex in examples[:3]:
            print(f"    '{ex['text']}' → {ex['expected']} ({ex['category']})")

    print(f"\nTotal examples: {len(test_sets['all'])}")
    print(f"Few-shot examples available: {len(FEW_SHOT_EXAMPLES)}")


if __name__ == "__main__":
    print_dataset_summary()