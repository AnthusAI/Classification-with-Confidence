"""
Fine-tuning simulation using few-shot learning to improve confidence on Gen Z slang.
"""
from typing import List, Dict, Any, Optional
import statistics
from classifier import OllamaSentimentClassifier
from confidence import ConfidenceScorer
from datasets import get_few_shot_examples, get_examples_by_category


class FewShotLearning:
    """
    Simulates fine-tuning through few-shot learning examples.
    """

    def __init__(self, classifier: OllamaSentimentClassifier):
        """
        Initialize the few-shot learning system.

        Args:
            classifier: The sentiment classifier to use
        """
        self.classifier = classifier
        self.confidence_scorer = ConfidenceScorer(classifier)

    def compare_before_after_learning(self, test_examples: List[Dict[str, Any]],
                                    few_shot_examples: List[Dict[str, str]],
                                    num_samples: int = 5) -> Dict[str, Any]:
        """
        Compare classification confidence before and after few-shot learning.

        Args:
            test_examples: Examples to test on (with 'text' and 'expected' keys)
            few_shot_examples: Examples to use for few-shot learning
            num_samples: Number of classification samples per test

        Returns:
            Comparison results showing improvement
        """
        print("🔬 Running Before/After Few-Shot Learning Experiment")
        print("=" * 60)

        # Test WITHOUT few-shot examples (baseline)
        print("\n📊 BEFORE Few-Shot Learning:")
        print("-" * 30)

        before_results = []
        for example in test_examples:
            result = self.confidence_scorer.classify_with_confidence(
                example['text'], num_samples=num_samples
            )
            result.update({
                'text': example['text'],
                'expected': example['expected'],
                'category': example.get('category', 'unknown')
            })
            before_results.append(result)

        # Test WITH few-shot examples
        print(f"\n📚 AFTER Few-Shot Learning ({len(few_shot_examples)} examples):")
        print("-" * 30)

        after_results = []
        for example in test_examples:
            result = self.confidence_scorer.classify_with_confidence(
                example['text'], num_samples=num_samples,
                few_shot_examples=few_shot_examples
            )
            result.update({
                'text': example['text'],
                'expected': example['expected'],
                'category': example.get('category', 'unknown')
            })
            after_results.append(result)

        # Compare results
        return self._analyze_improvement(before_results, after_results, few_shot_examples)

    def _analyze_improvement(self, before_results: List[Dict[str, Any]],
                           after_results: List[Dict[str, Any]],
                           few_shot_examples: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Analyze the improvement from few-shot learning.

        Args:
            before_results: Results before few-shot learning
            after_results: Results after few-shot learning
            few_shot_examples: The few-shot examples used

        Returns:
            Analysis of improvement
        """
        print("\n📈 IMPROVEMENT ANALYSIS:")
        print("=" * 30)

        improvements = []
        accuracy_before = 0
        accuracy_after = 0
        valid_comparisons = 0

        for before, after in zip(before_results, after_results):
            if before['prediction'] is not None and after['prediction'] is not None:
                # Calculate confidence improvement
                conf_improvement = after['confidence'] - before['confidence']
                improvements.append(conf_improvement)

                # Track accuracy
                accuracy_before += int(before['prediction'] == before['expected'])
                accuracy_after += int(after['prediction'] == after['expected'])
                valid_comparisons += 1

                # Show individual results
                print(f"\nText: '{before['text'][:50]}{'...' if len(before['text']) > 50 else ''}'")
                print(f"  Expected: {before['expected']}")
                print(f"  Before: {before['prediction']} (conf: {before['confidence']:.2f})")
                print(f"  After:  {after['prediction']} (conf: {after['confidence']:.2f})")
                print(f"  Confidence Δ: {conf_improvement:+.2f}")

        # Calculate overall statistics
        avg_improvement = statistics.mean(improvements) if improvements else 0.0
        accuracy_before_pct = accuracy_before / valid_comparisons if valid_comparisons > 0 else 0.0
        accuracy_after_pct = accuracy_after / valid_comparisons if valid_comparisons > 0 else 0.0

        # Group by category for analysis
        category_analysis = self._analyze_by_category(before_results, after_results)

        summary = {
            'avg_confidence_improvement': avg_improvement,
            'accuracy_before': accuracy_before_pct,
            'accuracy_after': accuracy_after_pct,
            'accuracy_improvement': accuracy_after_pct - accuracy_before_pct,
            'num_examples_tested': len(before_results),
            'valid_comparisons': valid_comparisons,
            'few_shot_examples_used': len(few_shot_examples),
            'category_analysis': category_analysis,
            'individual_improvements': improvements
        }

        self._print_summary(summary)
        return summary

    def _analyze_by_category(self, before_results: List[Dict[str, Any]],
                           after_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Analyze improvement by example category.

        Args:
            before_results: Results before few-shot learning
            after_results: Results after few-shot learning

        Returns:
            Category-wise analysis
        """
        categories = {}

        for before, after in zip(before_results, after_results):
            category = before.get('category', 'unknown')

            if category not in categories:
                categories[category] = {
                    'confidence_before': [],
                    'confidence_after': [],
                    'accuracy_before': [],
                    'accuracy_after': []
                }

            if before['prediction'] is not None and after['prediction'] is not None:
                categories[category]['confidence_before'].append(before['confidence'])
                categories[category]['confidence_after'].append(after['confidence'])
                categories[category]['accuracy_before'].append(
                    int(before['prediction'] == before['expected'])
                )
                categories[category]['accuracy_after'].append(
                    int(after['prediction'] == after['expected'])
                )

        # Calculate statistics for each category
        category_stats = {}
        for category, data in categories.items():
            if data['confidence_before']:
                category_stats[category] = {
                    'avg_confidence_before': statistics.mean(data['confidence_before']),
                    'avg_confidence_after': statistics.mean(data['confidence_after']),
                    'confidence_improvement': statistics.mean(data['confidence_after']) - statistics.mean(data['confidence_before']),
                    'accuracy_before': statistics.mean(data['accuracy_before']),
                    'accuracy_after': statistics.mean(data['accuracy_after']),
                    'accuracy_improvement': statistics.mean(data['accuracy_after']) - statistics.mean(data['accuracy_before']),
                    'count': len(data['confidence_before'])
                }

        return category_stats

    def _print_summary(self, summary: Dict[str, Any]):
        """
        Print a formatted summary of the improvement analysis.

        Args:
            summary: Summary statistics dictionary
        """
        print(f"\n📊 OVERALL SUMMARY:")
        print("=" * 30)
        print(f"Examples tested: {summary['num_examples_tested']}")
        print(f"Valid comparisons: {summary['valid_comparisons']}")
        print(f"Few-shot examples used: {summary['few_shot_examples_used']}")
        print()
        print(f"Average confidence improvement: {summary['avg_confidence_improvement']:+.3f}")
        print(f"Accuracy before: {summary['accuracy_before']:.1%}")
        print(f"Accuracy after: {summary['accuracy_after']:.1%}")
        print(f"Accuracy improvement: {summary['accuracy_improvement']:+.1%}")

        print(f"\n📋 BY CATEGORY:")
        print("-" * 20)
        for category, stats in summary['category_analysis'].items():
            print(f"\n{category.upper()} ({stats['count']} examples):")
            print(f"  Confidence: {stats['avg_confidence_before']:.2f} → {stats['avg_confidence_after']:.2f} ({stats['confidence_improvement']:+.2f})")
            print(f"  Accuracy: {stats['accuracy_before']:.1%} → {stats['accuracy_after']:.1%} ({stats['accuracy_improvement']:+.1%})")

    def demonstrate_few_shot_impact(self):
        """
        Run a comprehensive demonstration of few-shot learning impact.
        """
        print("🚀 Few-Shot Learning Impact Demonstration")
        print("=" * 50)

        # Get test examples - focus on Gen Z slang that should benefit from few-shot learning
        gen_z_examples = (
            get_examples_by_category('gen_z_positive')[:3] +
            get_examples_by_category('gen_z_negative')[:3]
        )

        # Also test some standard examples as a control group
        standard_examples = (
            get_examples_by_category('obvious_positive')[:2] +
            get_examples_by_category('obvious_negative')[:2]
        )

        all_test_examples = gen_z_examples + standard_examples

        # Get few-shot examples
        few_shot_examples = get_few_shot_examples()

        print(f"\nTesting {len(all_test_examples)} examples:")
        for ex in all_test_examples:
            print(f"  - '{ex['text'][:40]}...' ({ex['category']})")

        print(f"\nUsing {len(few_shot_examples)} few-shot examples:")
        for ex in few_shot_examples:
            print(f"  - '{ex['text']}' → {ex['sentiment']}")

        # Run the comparison
        results = self.compare_before_after_learning(
            all_test_examples, few_shot_examples, num_samples=3
        )

        return results


def main():
    """
    Test the few-shot learning functionality.
    """
    # Initialize classifier
    classifier = OllamaSentimentClassifier()

    # Test connection
    if not classifier.test_connection():
        print("❌ Failed to connect to Ollama. Make sure it's running!")
        return

    # Run few-shot learning demonstration
    few_shot_learner = FewShotLearning(classifier)
    few_shot_learner.demonstrate_few_shot_impact()


if __name__ == "__main__":
    main()