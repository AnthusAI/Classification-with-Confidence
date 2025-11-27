#!/usr/bin/env python3
"""
Compare Base vs Fine-Tuned Model Performance and Confidence Calibration

This script evaluates both the base Llama 3.1 model and the fine-tuned version
on our sentiment classification dataset, comparing accuracy, calibration metrics,
and confidence distributions.

Usage:
    python compare_base_vs_finetuned.py

Requirements:
    - fine_tune_model.py must have been run first
    - All existing confidence scoring dependencies
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import os
import numpy as np
import json
import random
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import time

# Import our existing modules
from classification_config import ClassificationConfig, ClassificationMode
from dataset_loader import DatasetLoader
from logprobs_confidence import TransformerLogprobsClassifier
# Removed FineTunedSentimentClassifier - using TransformerLogprobsClassifier for both (DRY)
from calibration import IsotonicRegressionCalibrator, PlattScalingCalibrator
from calibration_metrics import calibration_metrics, plot_reliability_diagram


class ModelComparator:
    """
    Compare base and fine-tuned models on accuracy and calibration metrics.
    """

    def __init__(self,
                 base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 fine_tuned_path: str = "./fine_tuned_sentiment_model"):
        """
        Initialize the comparator.

        Args:
            base_model_name: Base model name
            fine_tuned_path: Path to fine-tuned model
        """
        self.base_model_name = base_model_name
        self.fine_tuned_path = fine_tuned_path

        # Initialize models
        self.base_classifier = None
        self.fine_tuned_classifier = None

        # Results storage
        self.results = {
            'base': {'predictions': [], 'confidences': [], 'accuracies': []},
            'fine_tuned': {'predictions': [], 'confidences': [], 'accuracies': []}
        }

        # Create output directory
        os.makedirs("images/fine_tuning", exist_ok=True)

    def load_models(self):
        """Load both base and fine-tuned models."""
        print("Loading models...")

        # Load base model
        print("Loading base model...")
        self.base_classifier = TransformerLogprobsClassifier(self.base_model_name)

        # Load fine-tuned model using the SAME class (DRY principle)
        print("Loading fine-tuned model...")
        if not os.path.exists(self.fine_tuned_path):
            raise FileNotFoundError(f"Fine-tuned model not found at {self.fine_tuned_path}")

        self.fine_tuned_classifier = TransformerLogprobsClassifier(
            model_name=self.base_model_name,
            fine_tuned_path=self.fine_tuned_path
        )

        print("✅ Both models loaded successfully!")

    def evaluate_model(self, classifier, model_name: str, examples: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate a model on the test dataset.

        Args:
            classifier: Model classifier instance
            model_name: Name for logging
            examples: Test examples

        Returns:
            Evaluation results
        """
        print(f"Evaluating {model_name} model on {len(examples)} examples...")

        predictions = []
        confidences = []
        accuracies = []
        texts = []
        true_labels = []

        start_time = time.time()

        for i, example in enumerate(examples):
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(examples) - i - 1) / rate
                print(f"  Progress: {i + 1}/{len(examples)} ({rate:.1f} examples/sec, ETA: {eta:.0f}s)")

            text = example['text']
            true_label = example['expected']
            category = example.get('category', 'unknown')

            try:
                # Both models now use the same confidence calculation method
                result = classifier.get_real_logprobs_confidence(text)
                prediction = result.get('prediction', 'positive')
                confidence = result.get('confidence', 0.5)

                # Calculate accuracy
                accuracy = 1 if prediction == true_label else 0

                predictions.append(prediction)
                confidences.append(confidence)
                accuracies.append(accuracy)
                texts.append(text)
                true_labels.append(true_label)

            except Exception as e:
                print(f"  Error processing example {i}: {e}")
                # Use default values for failed examples
                predictions.append('positive')
                confidences.append(0.5)
                accuracies.append(0)
                texts.append(text)
                true_labels.append(true_label)

        total_time = time.time() - start_time
        rate = len(examples) / total_time

        print(f"  Completed in {total_time:.1f}s ({rate:.1f} examples/sec)")

        return {
            'predictions': predictions,
            'confidences': np.array(confidences),
            'accuracies': np.array(accuracies),
            'texts': texts,
            'true_labels': true_labels,
            'total_time': total_time,
            'examples_per_second': rate
        }

    def run_comparison(self, max_examples: int = 1000):
        """
        Run the full comparison between base and fine-tuned models.

        Args:
            max_examples: Number of examples to use for evaluation (default: 1000 for comprehensive analysis)
        """
        print("🎯 Starting Base vs Fine-Tuned Model Comparison")
        print("=" * 60)

        # Load models
        self.load_models()

        # CRITICAL: Load held-out test set for proper evaluation
        print("📂 Loading dataset...")
        
        # First, try to load the saved test set from fine-tuning
        test_set_path = "fine_tuned_sentiment_model/test_set.json"
        if os.path.exists(test_set_path):
            print("✅ Loading held-out test set from fine-tuning (proper evaluation)")
            with open(test_set_path, 'r') as f:
                examples = json.load(f)
            print(f"📊 Test set examples: {len(examples)} (20% held-out from training)")
        else:
            print("⚠️  No saved test set found - using random sample (not ideal for evaluation)")
            loader = DatasetLoader()
            all_examples = loader.load_all()
            print(f"📊 Total examples available: {len(all_examples)}")
            
            # Convert to the expected format and ensure proper random sampling
            examples = []
            for item in all_examples:
                examples.append({
                    'text': item['text'],
                    'expected': item['expected'],
                    'category': item.get('category', 'unknown')
                })
            
            # CRITICAL: Random shuffle to ensure diverse sampling
            random.seed(42)  # For reproducibility
            random.shuffle(examples)
        
        # Sample the requested number of examples (if test set is larger than requested)
        if max_examples and max_examples < len(examples):
            # For test set, take first N examples (already properly split)
            examples = examples[:max_examples]
            print(f"📝 Using {len(examples)} examples from test set")
            
            # Show category distribution of sample
            category_counts = {}
            for ex in examples:
                cat = ex['category']
                category_counts[cat] = category_counts.get(cat, 0) + 1
            
            print("📊 Sample distribution by category:")
            for cat, count in sorted(category_counts.items()):
                print(f"  {cat}: {count} examples")
        else:
            print(f"📝 Using all {len(examples)} examples")

        # Evaluate base model
        print("\n📊 Evaluating Base Model...")
        base_results = self.evaluate_model(self.base_classifier, "Base", examples)

        # Evaluate fine-tuned model
        print("\n📊 Evaluating Fine-Tuned Model...")
        fine_tuned_results = self.evaluate_model(self.fine_tuned_classifier, "Fine-Tuned", examples)

        # Store results
        self.results['base'] = base_results
        self.results['fine_tuned'] = fine_tuned_results

        # Calculate metrics
        print("\n📈 Calculating Metrics...")
        base_metrics = self.calculate_metrics(base_results, "Base")
        fine_tuned_metrics = self.calculate_metrics(fine_tuned_results, "Fine-Tuned")

        # Print comparison
        self.print_comparison(base_metrics, fine_tuned_metrics)

        # Create visualizations
        print("\n📊 Creating Visualizations...")
        self.create_visualizations(base_results, fine_tuned_results)

        # Save detailed results
        self.save_results(base_metrics, fine_tuned_metrics)

        print("\n✅ Comparison completed!")

    def calculate_metrics(self, results: Dict, model_name: str) -> Dict[str, Any]:
        """Calculate comprehensive metrics for a model."""

        confidences = results['confidences']
        accuracies = results['accuracies']

        # Basic metrics
        accuracy = accuracies.mean()
        mean_confidence = confidences.mean()

        # Calibration metrics
        cal_metrics = calibration_metrics(confidences, accuracies)

        # Confidence threshold analysis
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        threshold_analysis = {}

        for threshold in thresholds:
            mask = confidences >= threshold
            if mask.sum() > 0:
                threshold_accuracy = accuracies[mask].mean()
                threshold_count = mask.sum()
            else:
                threshold_accuracy = 0.0
                threshold_count = 0

            threshold_analysis[threshold] = {
                'accuracy': threshold_accuracy,
                'count': threshold_count,
                'percentage': threshold_count / len(confidences)
            }

        # Confidence distribution
        conf_distribution = {
            'mean': mean_confidence,
            'std': confidences.std(),
            'median': np.median(confidences),
            'min': confidences.min(),
            'max': confidences.max(),
            'q25': np.percentile(confidences, 25),
            'q75': np.percentile(confidences, 75)
        }

        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
            'calibration_metrics': cal_metrics,
            'threshold_analysis': threshold_analysis,
            'confidence_distribution': conf_distribution,
            'total_examples': len(confidences),
            'performance': {
                'total_time': results.get('total_time', 0),
                'examples_per_second': results.get('examples_per_second', 0)
            }
        }

    def print_comparison(self, base_metrics: Dict, fine_tuned_metrics: Dict):
        """Print detailed comparison results."""

        print("\n📊 DETAILED COMPARISON RESULTS")
        print("=" * 70)

        # Overall performance
        print(f"\n🎯 Overall Performance:")
        print(f"{'Metric':<25} {'Base Model':<15} {'Fine-Tuned':<15} {'Improvement':<15}")
        print("-" * 70)

        base_acc = base_metrics['accuracy']
        ft_acc = fine_tuned_metrics['accuracy']
        acc_improvement = ft_acc - base_acc

        base_conf = base_metrics['mean_confidence']
        ft_conf = fine_tuned_metrics['mean_confidence']
        conf_change = ft_conf - base_conf

        print(f"{'Accuracy':<25} {base_acc:<15.3f} {ft_acc:<15.3f} {acc_improvement:<15.3f}")
        print(f"{'Mean Confidence':<25} {base_conf:<15.3f} {ft_conf:<15.3f} {conf_change:<15.3f}")

        # Calibration metrics
        print(f"\n📏 Calibration Metrics:")
        base_cal = base_metrics['calibration_metrics']
        ft_cal = fine_tuned_metrics['calibration_metrics']

        ece_improvement = base_cal['ECE'] - ft_cal['ECE']
        mce_improvement = base_cal['MCE'] - ft_cal['MCE']

        print(f"{'ECE (Expected)':<25} {base_cal['ECE']:<15.3f} {ft_cal['ECE']:<15.3f} {ece_improvement:<15.3f}")
        print(f"{'MCE (Maximum)':<25} {base_cal['MCE']:<15.3f} {ft_cal['MCE']:<15.3f} {mce_improvement:<15.3f}")

        # Threshold analysis
        print(f"\n🎯 High-Confidence Predictions (Threshold ≥ 90%):")
        base_90 = base_metrics['threshold_analysis'][0.9]
        ft_90 = fine_tuned_metrics['threshold_analysis'][0.9]

        print(f"{'Count':<25} {base_90['count']:<15} {ft_90['count']:<15} {ft_90['count'] - base_90['count']:<15}")
        print(f"{'Accuracy':<25} {base_90['accuracy']:<15.3f} {ft_90['accuracy']:<15.3f} {ft_90['accuracy'] - base_90['accuracy']:<15.3f}")
        print(f"{'Percentage':<25} {base_90['percentage']:<15.3f} {ft_90['percentage']:<15.3f} {ft_90['percentage'] - base_90['percentage']:<15.3f}")

        # Performance
        print(f"\n⚡ Performance:")
        base_perf = base_metrics['performance']
        ft_perf = fine_tuned_metrics['performance']

        print(f"{'Examples/second':<25} {base_perf['examples_per_second']:<15.1f} {ft_perf['examples_per_second']:<15.1f}")
        print(f"{'Total time (s)':<25} {base_perf['total_time']:<15.1f} {ft_perf['total_time']:<15.1f}")

    def create_visualizations(self, base_results: Dict, fine_tuned_results: Dict):
        """Create comparison visualizations."""

        # 1. Reliability diagrams comparison
        self.create_reliability_comparison(base_results, fine_tuned_results)

        # 2. Confidence distribution comparison
        self.create_confidence_distribution_comparison(base_results, fine_tuned_results)

        # 3. Threshold analysis
        self.create_threshold_analysis(base_results, fine_tuned_results)

        # 4. Performance by category
        self.create_category_analysis(base_results, fine_tuned_results)

    def create_reliability_comparison(self, base_results: Dict, fine_tuned_results: Dict):
        """Create side-by-side reliability diagrams."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Base model reliability
        plot_reliability_diagram(
            base_results['confidences'],
            base_results['accuracies'],
            ax=ax1,
            title="Base Model Reliability",
            show_ece=True
        )

        # Fine-tuned model reliability
        plot_reliability_diagram(
            fine_tuned_results['confidences'],
            fine_tuned_results['accuracies'],
            ax=ax2,
            title="Fine-Tuned Model Reliability",
            show_ece=True
        )

        fig.suptitle('Base vs Fine-Tuned Model Calibration Comparison', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig("images/fine_tuning/calibration_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✅ Calibration comparison saved")

    def create_confidence_distribution_comparison(self, base_results: Dict, fine_tuned_results: Dict):
        """Create confidence distribution comparison."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Base model distribution
        ax1.hist(base_results['confidences'], bins=30, alpha=0.7, color='red', edgecolor='black')
        ax1.set_title('Base Model\nConfidence Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Number of Predictions', fontsize=12)
        ax1.axvline(np.mean(base_results['confidences']), color='darkred', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(base_results["confidences"]):.3f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fine-tuned model distribution
        ax2.hist(fine_tuned_results['confidences'], bins=30, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Fine-Tuned Model\nConfidence Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence Score', fontsize=12)
        ax2.set_ylabel('Number of Predictions', fontsize=12)
        ax2.axvline(np.mean(fine_tuned_results['confidences']), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(fine_tuned_results["confidences"]):.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle('Confidence Distribution Changes After Fine-Tuning', fontsize=16, fontweight='bold')

        plt.tight_layout()
        plt.savefig("images/fine_tuning/confidence_distribution_changes.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✅ Confidence distribution comparison saved")

    def create_threshold_analysis(self, base_results: Dict, fine_tuned_results: Dict):
        """Create threshold analysis visualization with business decision buckets."""

        thresholds = np.arange(0.5, 1.0, 0.05)

        base_accuracies = []
        base_counts = []
        ft_accuracies = []
        ft_counts = []

        for threshold in thresholds:
            # Base model
            base_mask = base_results['confidences'] >= threshold
            if base_mask.sum() > 0:
                base_accuracies.append(base_results['accuracies'][base_mask].mean())
                base_counts.append(base_mask.sum())
            else:
                base_accuracies.append(0)
                base_counts.append(0)

            # Fine-tuned model
            ft_mask = fine_tuned_results['confidences'] >= threshold
            if ft_mask.sum() > 0:
                ft_accuracies.append(fine_tuned_results['accuracies'][ft_mask].mean())
                ft_counts.append(ft_mask.sum())
            else:
                ft_accuracies.append(0)
                ft_counts.append(0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Add colored background regions for business decision zones
        for ax in [ax1, ax2]:
            ax.axvspan(90, 100, alpha=0.15, color='green', label='High Confidence Zone (≥90%)')
            ax.axvspan(80, 90, alpha=0.15, color='yellow', label='Medium-High Zone (80-90%)')
            ax.axvspan(70, 80, alpha=0.15, color='orange', label='Medium Zone (70-80%)')
            ax.axvspan(50, 70, alpha=0.15, color='red', label='Low Confidence Zone (50-70%)')

        # Accuracy at thresholds
        ax1.plot(thresholds * 100, base_accuracies, 'r-', linewidth=4, label='Base Model', alpha=0.9, zorder=5)
        ax1.plot(thresholds * 100, ft_accuracies, 'g-', linewidth=4, label='Fine-Tuned Model', alpha=0.9, zorder=5)
        ax1.plot([50, 100], [0.5, 1.0], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration', zorder=4)

        # Add key business threshold lines
        ax1.axvline(90, color='purple', linestyle='--', linewidth=3, alpha=0.8, zorder=3)
        ax1.axhline(90, color='purple', linestyle='--', linewidth=3, alpha=0.8, zorder=3)

        # Add annotations for key business thresholds
        key_business_thresholds = [70, 80, 90, 95]
        for threshold_pct in key_business_thresholds:
            threshold = threshold_pct / 100
            threshold_idx = np.argmin(np.abs(thresholds - threshold))
            if threshold_idx < len(thresholds) and ft_counts[threshold_idx] > 0:
                ax1.annotate(f'{threshold_pct}% threshold:\n{ft_counts[threshold_idx]} samples\n{ft_accuracies[threshold_idx]:.1%} accurate', 
                           xy=(threshold_pct, ft_accuracies[threshold_idx] * 100),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                           fontsize=9, ha='left')

        ax1.set_xlabel('Confidence Threshold (%)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy at Threshold (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Business Decision Thresholds: Accuracy vs Confidence\n' +
                     'Each colored region represents a business decision bucket', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3, zorder=1)
        ax1.set_ylim(0, 100)

        # Sample counts at thresholds
        ax2.plot(thresholds * 100, base_counts, 'r-', linewidth=4, label='Base Model', alpha=0.9, zorder=5)
        ax2.plot(thresholds * 100, ft_counts, 'g-', linewidth=4, label='Fine-Tuned Model', alpha=0.9, zorder=5)

        # Add business threshold lines
        for threshold_pct in key_business_thresholds:
            ax2.axvline(threshold_pct, color='purple', linestyle='--', linewidth=2, alpha=0.6, zorder=3)

        ax2.set_xlabel('Confidence Threshold (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Predictions Above Threshold', fontsize=14, fontweight='bold')
        ax2.set_title('Sample Volume at Business Decision Thresholds\n' +
                     'Shows how many predictions fall into each confidence bucket', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, zorder=1)

        # Add business interpretation text
        business_text = ("Business Impact:\n" +
                        "• Green Zone: Auto-approve\n" +
                        "• Yellow Zone: Review recommended\n" +
                        "• Orange Zone: Manual check required\n" +
                        "• Red Zone: High uncertainty")
        ax2.text(0.02, 0.98, business_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.9, edgecolor='navy'),
                fontsize=10, verticalalignment='top', fontweight='bold')

        plt.tight_layout()
        plt.savefig("images/fine_tuning/threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✅ Threshold analysis with business buckets saved")

    def create_category_analysis(self, base_results: Dict, fine_tuned_results: Dict):
        """Analyze performance by category."""

        # Get examples with categories using new dataset loader
        loader = DatasetLoader()
        all_examples = loader.load_all()
        
        # Convert to expected format
        examples = []
        for item in all_examples:
            examples.append({
                'text': item['text'],
                'expected': item['expected'],
                'category': item.get('category', 'unknown')
            })

        # Group by category
        category_stats = defaultdict(lambda: {'base': [], 'fine_tuned': [], 'texts': []})

        for i, example in enumerate(examples):
            if i >= len(base_results['accuracies']):
                break

            category = example.get('category', 'unknown')
            category_stats[category]['base'].append(base_results['accuracies'][i])
            category_stats[category]['fine_tuned'].append(fine_tuned_results['accuracies'][i])
            category_stats[category]['texts'].append(example['text'])

        # Create visualization
        categories = sorted(category_stats.keys())
        base_accs = [np.mean(category_stats[cat]['base']) for cat in categories]
        ft_accs = [np.mean(category_stats[cat]['fine_tuned']) for cat in categories]

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax.bar(x - width/2, base_accs, width, label='Base Model', alpha=0.8, color='red')
        bars2 = ax.bar(x + width/2, ft_accs, width, label='Fine-Tuned Model', alpha=0.8, color='green')

        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by Category: Base vs Fine-Tuned', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig("images/fine_tuning/category_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("  ✅ Category analysis saved")

    def save_results(self, base_metrics: Dict, fine_tuned_metrics: Dict):
        """Save detailed results to JSON."""

        results_summary = {
            'comparison_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'base_model': {
                'name': self.base_model_name,
                'metrics': base_metrics
            },
            'fine_tuned_model': {
                'path': self.fine_tuned_path,
                'metrics': fine_tuned_metrics
            },
            'improvements': {
                'accuracy': fine_tuned_metrics['accuracy'] - base_metrics['accuracy'],
                'ece_reduction': base_metrics['calibration_metrics']['ECE'] - fine_tuned_metrics['calibration_metrics']['ECE'],
                'mce_reduction': base_metrics['calibration_metrics']['MCE'] - fine_tuned_metrics['calibration_metrics']['MCE']
            }
        }

        with open('images/fine_tuning/comparison_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        print("  ✅ Results saved to comparison_results.json")


def main():
    """Run the comparison."""
    print("🔍 Base vs Fine-Tuned Model Comparison")
    print("=" * 50)

    # Check if fine-tuned model exists
    fine_tuned_path = "./fine_tuned_sentiment_model"
    if not os.path.exists(fine_tuned_path):
        print(f"❌ Fine-tuned model not found at {fine_tuned_path}")
        print("Please run the following steps first:")
        print("1. pip install peft bitsandbytes datasets accelerate")
        print("2. python fine_tune_model.py")
        return

    # Initialize comparator
    comparator = ModelComparator(fine_tuned_path=fine_tuned_path)

    # Run comparison with 1000 examples for comprehensive analysis
    try:
        comparator.run_comparison(max_examples=1000)  # Full evaluation with realistic business numbers

        print("\n🎉 Comparison completed successfully!")
        print("📊 Check the images/fine_tuning/ directory for visualizations")

    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()