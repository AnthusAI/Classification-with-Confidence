#!/usr/bin/env python3
"""
Calibration Binning Visualization

This script creates a detailed visualization showing how ECE binning works
and why it can be misleading compared to visual calibration assessment.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from datasets import get_test_sets
from calibration import IsotonicRegressionCalibrator
from calibration_metrics import expected_calibration_error, maximum_calibration_error
import matplotlib.patches as patches

def create_binning_explanation():
    """Create a detailed visualization of how ECE binning works."""
    
    print("📊 Creating ECE binning explanation visualization...")
    
    # Load real data and simulate confidence patterns
    test_sets = get_test_sets()
    all_examples = test_sets['all']
    
    np.random.seed(42)
    
    confidences = []
    accuracies = []
    
    # Use all 1000 samples to match our main experiments
    for example in all_examples[:1000]:
        category = example.get('category', 'unknown')
        
        if category in ['strongly_positive', 'strongly_negative']:
            true_difficulty = np.random.beta(1, 9)
            confidence = 0.85 + 0.14 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.05 - 0.05 * np.random.random()
        elif category in ['mildly_positive', 'mildly_negative']:
            true_difficulty = np.random.beta(2, 6)
            confidence = 0.75 + 0.20 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.08 - 0.07 * np.random.random()
        elif category in ['weakly_positive', 'weakly_negative']:
            true_difficulty = np.random.beta(3, 3)
            confidence = 0.55 + 0.25 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.12 - 0.13 * np.random.random()
        elif category in ['ambiguous_positive', 'ambiguous_negative']:
            true_difficulty = np.random.beta(5, 2)
            confidence = 0.45 + 0.25 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.20 - 0.15 * np.random.random()
        elif category == 'sarcastic':
            true_difficulty = np.random.beta(4, 3)
            confidence = 0.50 + 0.25 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.15 - 0.10 * np.random.random()
        else:
            true_difficulty = np.random.beta(3, 3)
            confidence = 0.55 + 0.30 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.10 - 0.10 * np.random.random()
        
        accuracy_prob = max(0.1, min(0.95, accuracy_prob))
        pred_correct = np.random.random() < accuracy_prob
        
        confidences.append(confidence)
        accuracies.append(1 if pred_correct else 0)
    
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    # Train calibrator
    isotonic_cal = IsotonicRegressionCalibrator()
    isotonic_cal.fit(confidences, accuracies)
    calibrated_confidences = isotonic_cal.predict_proba(confidences)
    
    # Calculate both metrics with 20 bins for finer granularity
    raw_ece = expected_calibration_error(confidences, accuracies, n_bins=20)
    raw_mce = maximum_calibration_error(confidences, accuracies, n_bins=20)
    cal_ece = expected_calibration_error(calibrated_confidences, accuracies, n_bins=20)
    cal_mce = maximum_calibration_error(calibrated_confidences, accuracies, n_bins=20)
    
    # Create the visualization - focus on the key concept
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Raw confidence with bins
    plot_reliability_with_bins(ax1, confidences, accuracies, "Raw Confidence", 
                              f"ECE = {raw_ece:.3f}, MCE = {raw_mce:.3f}")
    
    # Right: Calibrated confidence with bins
    plot_reliability_with_bins(ax2, calibrated_confidences, accuracies, "Calibrated Confidence", 
                              f"ECE = {cal_ece:.3f}, MCE = {cal_mce:.3f}")
    
    # Add overall explanation
    fig.suptitle('How ECE Binning Works: Why "Perfect" ECE Can Be Misleading', 
                 fontsize=16, fontweight='bold')
    
    explanation_text = (
        "Key Insight: ECE divides predictions into 20 bins (0-5%, 5-10%, etc.) for finer granularity.\n"
        "• Blue dots = individual predictions (can be poorly calibrated)\n"
        "• Red dots = bin averages (what ECE actually measures)\n"
        "• Missing bins = no predictions in that confidence range\n"
        "• More bins = more detailed calibration assessment, may reveal hidden problems"
    )
    
    fig.text(0.5, 0.02, explanation_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig("images/calibration/calibration_binning_explanation.png", dpi=300, bbox_inches='tight')
    print("💾 Calibration binning explanation saved")
    plt.close()
    
    return {
        'raw_ece': raw_ece,
        'raw_mce': raw_mce,
        'cal_ece': cal_ece,
        'cal_mce': cal_mce
    }

def plot_reliability_with_bins(ax, confidences, accuracies, title, metrics_text):
    """Plot reliability diagram with visible bins."""
    
    n_bins = 20
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
    
    # Plot bins as rectangles and calculate bin statistics
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        # Find predictions in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_center = (bin_lower + bin_upper) / 2
            bin_accuracy = accuracies[in_bin].mean()
            bin_count = in_bin.sum()
            
            bin_centers.append(bin_center)
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(bin_count)
            
            # Draw bin rectangle
            rect = patches.Rectangle((bin_lower, 0), bin_upper - bin_lower, 1, 
                                   linewidth=1, edgecolor='gray', facecolor='lightblue', alpha=0.1)
            ax.add_patch(rect)
            
            # Plot bin average as a large dot
            ax.plot(bin_center, bin_accuracy, 'ro', markersize=12, alpha=0.8, 
                   label='Bin Average' if i == 0 else "")
            
            # Add bin count label
            ax.text(bin_center, 0.05, f'{bin_count}', ha='center', fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Plot individual predictions as small dots
    ax.scatter(confidences, accuracies, alpha=0.3, s=20, c='blue', 
              label='Individual Predictions')
    
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'{title}\n{metrics_text}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main():
    """Create the binning explanation visualization."""
    
    print("🔍 CALIBRATION BINNING EXPLANATION")
    print("=" * 50)
    
    import os
    os.makedirs("images/calibration", exist_ok=True)
    
    metrics = create_binning_explanation()
    
    print(f"\n📊 METRICS COMPARISON:")
    print(f"Raw Confidence:")
    print(f"  ECE (Expected): {metrics['raw_ece']:.3f}")
    print(f"  MCE (Maximum):  {metrics['raw_mce']:.3f}")
    print(f"\nCalibrated Confidence:")
    print(f"  ECE (Expected): {metrics['cal_ece']:.3f}")
    print(f"  MCE (Maximum):  {metrics['cal_mce']:.3f}")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"• ECE can be misleadingly low even with poor individual calibration")
    print(f"• MCE shows the worst-case bin error (more conservative)")
    print(f"• Visual inspection often reveals problems ECE misses")
    print(f"• For business decisions, consider both metrics + visual assessment")

if __name__ == "__main__":
    main()
