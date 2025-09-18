#!/usr/bin/env python3
"""
Create Demo Visualizations for Fine-Tuning Section

This script creates realistic demonstration visualizations for the fine-tuning
section of the README, showing what users can expect from the comparison.

Usage:
    python create_demo_visualizations.py
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import os
from typing import Dict, List, Tuple
from calibration_metrics import plot_reliability_diagram

def create_demo_directories():
    """Create necessary directories for demo images."""
    os.makedirs("images/fine_tuning", exist_ok=True)
    print("📁 Created images/fine_tuning directory")

def generate_realistic_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate realistic base and fine-tuned model performance data.

    Returns:
        base_confidences, base_accuracies, ft_confidences, ft_accuracies
    """
    np.random.seed(42)  # For reproducible demo results

    # Base model: overconfident, lower accuracy
    base_confidences = np.random.beta(3, 1, n_samples)  # Tends toward high confidence

    # Create accuracy that's correlated with confidence but systematically overconfident
    base_accuracies = []
    for conf in base_confidences:
        # Model is overconfident: actual accuracy is lower than confidence
        true_accuracy_prob = conf * 0.85 + np.random.normal(0, 0.1)  # 15% overconfident on average
        true_accuracy_prob = max(0.1, min(0.95, true_accuracy_prob))  # Clamp
        is_correct = np.random.random() < true_accuracy_prob
        base_accuracies.append(1 if is_correct else 0)

    base_accuracies = np.array(base_accuracies)

    # Fine-tuned model: better calibrated, higher accuracy
    ft_confidences = np.random.beta(2.5, 1.5, n_samples)  # Slightly more conservative

    ft_accuracies = []
    for conf in ft_confidences:
        # Fine-tuned model is better calibrated: accuracy closer to confidence
        true_accuracy_prob = conf * 0.95 + np.random.normal(0, 0.05)  # Only 5% overconfident
        true_accuracy_prob = max(0.1, min(0.95, true_accuracy_prob))  # Clamp
        is_correct = np.random.random() < true_accuracy_prob
        ft_accuracies.append(1 if is_correct else 0)

    ft_accuracies = np.array(ft_accuracies)

    return base_confidences, base_accuracies, ft_confidences, ft_accuracies

def create_custom_reliability_plot(ax, confidences, accuracies, title):
    """Create a custom reliability plot on the given axes."""
    from calibration_metrics import expected_calibration_error

    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    # Calculate bin statistics
    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

        if in_bin.sum() > 0:
            bin_center = (bin_lower + bin_upper) / 2
            bin_accuracy = accuracies[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_count = in_bin.sum()

            bin_centers.append(bin_center)
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(bin_count)

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')

    # Plot bin points
    if bin_centers:
        ax.scatter(bin_confidences, bin_accuracies, s=[c*2 for c in bin_counts],
                  alpha=0.7, c='red', edgecolors='black', label='Binned Results')

    # Calculate and display ECE
    ece = expected_calibration_error(confidences, accuracies, n_bins)

    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{title}\nECE = {ece:.3f}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

def create_calibration_comparison():
    """Create the main calibration comparison visualization."""
    print("📊 Creating calibration comparison...")

    base_confidences, base_accuracies, ft_confidences, ft_accuracies = generate_realistic_data()

    # Create individual plots and then combine them
    fig1 = plot_reliability_diagram(
        base_confidences,
        base_accuracies,
        title="Base Model Reliability (Overconfident)",
        show=False
    )
    fig1.savefig("images/fine_tuning/base_model_reliability.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plot_reliability_diagram(
        ft_confidences,
        ft_accuracies,
        title="Fine-Tuned Model Reliability (Well-Calibrated)",
        show=False
    )
    fig2.savefig("images/fine_tuning/finetuned_model_reliability.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Create a simple combined visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Create custom reliability plots
    create_custom_reliability_plot(ax1, base_confidences, base_accuracies, "Base Model\n(Overconfident)")
    create_custom_reliability_plot(ax2, ft_confidences, ft_accuracies, "Fine-Tuned Model\n(Well-Calibrated)")

    fig.suptitle('Base vs Fine-Tuned Model Calibration Comparison', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig("images/fine_tuning/calibration_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✅ Calibration comparison saved")

def create_confidence_distribution_changes():
    """Create confidence distribution comparison."""
    print("📊 Creating confidence distribution changes...")

    base_confidences, base_accuracies, ft_confidences, ft_accuracies = generate_realistic_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Base model distribution
    ax1.hist(base_confidences, bins=30, alpha=0.7, color='red', edgecolor='black')
    ax1.set_title('Base Model\nConfidence Distribution\n(Overconfident)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Number of Predictions', fontsize=12)
    ax1.axvline(np.mean(base_confidences), color='darkred', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(base_confidences):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Fine-tuned model distribution
    ax2.hist(ft_confidences, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Fine-Tuned Model\nConfidence Distribution\n(Better Calibrated)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Number of Predictions', fontsize=12)
    ax2.axvline(np.mean(ft_confidences), color='darkgreen', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(ft_confidences):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    fig.suptitle('Confidence Distribution Changes After Fine-Tuning', fontsize=16, fontweight='bold')

    # Add explanation
    explanation = (
        "Left: Base model shows overconfidence (many high confidence scores)\n"
        "Right: Fine-tuned model is more realistic (confidence better matches accuracy)"
    )
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=12, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig("images/fine_tuning/confidence_distribution_changes.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✅ Confidence distribution changes saved")

def create_threshold_analysis():
    """Create threshold analysis visualization."""
    print("📊 Creating threshold analysis...")

    base_confidences, base_accuracies, ft_confidences, ft_accuracies = generate_realistic_data()

    thresholds = np.arange(0.5, 1.0, 0.05)

    base_accuracies_at_threshold = []
    base_counts_at_threshold = []
    ft_accuracies_at_threshold = []
    ft_counts_at_threshold = []

    for threshold in thresholds:
        # Base model
        base_mask = base_confidences >= threshold
        if base_mask.sum() > 0:
            base_accuracies_at_threshold.append(base_accuracies[base_mask].mean())
            base_counts_at_threshold.append(base_mask.sum())
        else:
            base_accuracies_at_threshold.append(0)
            base_counts_at_threshold.append(0)

        # Fine-tuned model
        ft_mask = ft_confidences >= threshold
        if ft_mask.sum() > 0:
            ft_accuracies_at_threshold.append(ft_accuracies[ft_mask].mean())
            ft_counts_at_threshold.append(ft_mask.sum())
        else:
            ft_accuracies_at_threshold.append(0)
            ft_counts_at_threshold.append(0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Accuracy at thresholds
    ax1.plot(thresholds * 100, base_accuracies_at_threshold, 'r-', linewidth=3,
             label='Base Model', alpha=0.8)
    ax1.plot(thresholds * 100, ft_accuracies_at_threshold, 'g-', linewidth=3,
             label='Fine-Tuned Model', alpha=0.8)
    ax1.plot([50, 100], [0.5, 1.0], 'k--', alpha=0.5, label='Perfect Calibration')

    ax1.set_xlabel('Confidence Threshold (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy at Threshold', fontsize=12, fontweight='bold')
    ax1.set_title('Accuracy vs Confidence Threshold', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.5, 1.0)

    # Sample counts at thresholds
    ax2.plot(thresholds * 100, base_counts_at_threshold, 'r-', linewidth=3,
             label='Base Model', alpha=0.8)
    ax2.plot(thresholds * 100, ft_counts_at_threshold, 'g-', linewidth=3,
             label='Fine-Tuned Model', alpha=0.8)

    ax2.set_xlabel('Confidence Threshold (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Predictions Above Threshold', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Size at Each Threshold', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("images/fine_tuning/threshold_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✅ Threshold analysis saved")

def create_business_impact_comparison():
    """Create business impact visualization."""
    print("📊 Creating business impact comparison...")

    base_confidences, base_accuracies, ft_confidences, ft_accuracies = generate_realistic_data()

    # Calculate business metrics at 90% threshold
    base_90_mask = base_confidences >= 0.9
    ft_90_mask = ft_confidences >= 0.9

    base_90_count = base_90_mask.sum()
    base_90_accuracy = base_accuracies[base_90_mask].mean() if base_90_count > 0 else 0
    base_90_errors = base_90_count - (base_90_accuracy * base_90_count)

    ft_90_count = ft_90_mask.sum()
    ft_90_accuracy = ft_accuracies[ft_90_mask].mean() if ft_90_count > 0 else 0
    ft_90_errors = ft_90_count - (ft_90_accuracy * ft_90_count)

    # Create bar chart comparison
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    models = ['Base Model', 'Fine-Tuned Model']

    # Predictions above 90% threshold
    counts = [base_90_count, ft_90_count]
    bars1 = ax1.bar(models, counts, color=['red', 'green'], alpha=0.7)
    ax1.set_title('Predictions Above 90% Threshold', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Predictions', fontsize=12)
    ax1.grid(True, alpha=0.3)

    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                f'{int(count)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Accuracy of high-confidence predictions
    accuracies = [base_90_accuracy, ft_90_accuracy]
    bars2 = ax2.bar(models, accuracies, color=['red', 'green'], alpha=0.7)
    ax2.set_title('Accuracy of High-Confidence\nPredictions (≥90%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_ylim(0.8, 1.0)
    ax2.grid(True, alpha=0.3)

    for bar, acc in zip(bars2, accuracies):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Number of errors (false positives)
    errors = [base_90_errors, ft_90_errors]
    bars3 = ax3.bar(models, errors, color=['red', 'green'], alpha=0.7)
    ax3.set_title('False Positives in\nHigh-Confidence Predictions', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Number of Errors', fontsize=12)
    ax3.grid(True, alpha=0.3)

    for bar, error in zip(bars3, errors):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{int(error)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add overall improvement text
    improvement_pct = ((ft_90_count - base_90_count) / base_90_count * 100) if base_90_count > 0 else 0
    error_reduction_pct = ((base_90_errors - ft_90_errors) / base_90_errors * 100) if base_90_errors > 0 else 0

    fig.suptitle(f'Business Impact: +{improvement_pct:.0f}% More Automated Decisions, -{error_reduction_pct:.0f}% Fewer Errors',
                 fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig("images/fine_tuning/business_impact_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✅ Business impact comparison saved")

def print_demo_metrics():
    """Print the metrics that would be shown in the README."""
    print("\n📈 DEMO METRICS SUMMARY")
    print("=" * 50)

    base_confidences, base_accuracies, ft_confidences, ft_accuracies = generate_realistic_data()

    # Calculate actual metrics
    from calibration_metrics import calibration_metrics

    base_metrics = calibration_metrics(base_confidences, base_accuracies)
    ft_metrics = calibration_metrics(ft_confidences, ft_accuracies)

    base_accuracy = base_accuracies.mean()
    ft_accuracy = ft_accuracies.mean()

    # High confidence performance
    base_90_mask = base_confidences >= 0.9
    ft_90_mask = ft_confidences >= 0.9

    base_90_accuracy = base_accuracies[base_90_mask].mean() if base_90_mask.sum() > 0 else 0
    ft_90_accuracy = ft_accuracies[ft_90_mask].mean() if ft_90_mask.sum() > 0 else 0

    print(f"📊 Overall Performance:")
    print(f"  Base Model Accuracy:      {base_accuracy:.3f}")
    print(f"  Fine-Tuned Accuracy:      {ft_accuracy:.3f}")
    print(f"  Accuracy Improvement:     +{ft_accuracy - base_accuracy:.3f}")
    print()

    print(f"📏 Calibration Metrics:")
    print(f"  Base Model ECE:           {base_metrics['ECE']:.3f}")
    print(f"  Fine-Tuned ECE:           {ft_metrics['ECE']:.3f}")
    print(f"  ECE Improvement:          -{base_metrics['ECE'] - ft_metrics['ECE']:.3f}")
    print()

    print(f"🎯 High-Confidence Performance (≥90%):")
    print(f"  Base Model Count:         {base_90_mask.sum()}")
    print(f"  Fine-Tuned Count:         {ft_90_mask.sum()}")
    print(f"  Base Model Accuracy:      {base_90_accuracy:.3f}")
    print(f"  Fine-Tuned Accuracy:      {ft_90_accuracy:.3f}")
    print()

    print(f"💡 Business Impact:")
    if base_90_mask.sum() > 0:
        improvement = (ft_90_mask.sum() - base_90_mask.sum()) / base_90_mask.sum() * 100
        print(f"  More automated decisions: +{improvement:.0f}%")

    base_errors = base_90_mask.sum() - (base_90_accuracy * base_90_mask.sum())
    ft_errors = ft_90_mask.sum() - (ft_90_accuracy * ft_90_mask.sum())
    if base_errors > 0:
        error_reduction = (base_errors - ft_errors) / base_errors * 100
        print(f"  Fewer errors:             -{error_reduction:.0f}%")

def main():
    """Create all demo visualizations."""
    print("🎨 Creating Fine-Tuning Demo Visualizations")
    print("=" * 60)

    create_demo_directories()

    # Create all visualizations
    create_calibration_comparison()
    create_confidence_distribution_changes()
    create_threshold_analysis()
    create_business_impact_comparison()

    # Print summary metrics
    print_demo_metrics()

    print("\n✅ All demo visualizations created successfully!")
    print("📁 Check the images/fine_tuning/ directory for the generated charts")
    print("\n💡 These demonstrate the expected improvements from fine-tuning:")
    print("  • Better calibration (ECE reduction)")
    print("  • Higher accuracy on the task")
    print("  • More reliable high-confidence predictions")
    print("  • Better business decision-making metrics")

if __name__ == "__main__":
    main()