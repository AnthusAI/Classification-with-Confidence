#!/usr/bin/env python3
"""
Create individual reliability diagrams for each calibration method.
Shows how different calibration approaches move the dots closer to the perfect line.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from datasets import get_test_sets
from calibration import PlattScalingCalibrator, IsotonicRegressionCalibrator
from calibration_metrics import expected_calibration_error, maximum_calibration_error

def plot_single_reliability_diagram(ax, confidences, accuracies, title, method_name):
    """Plot a single reliability diagram with bins."""
    
    n_bins = 20
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Calculate ECE and MCE
    ece = expected_calibration_error(confidences, accuracies, n_bins=n_bins)
    mce = maximum_calibration_error(confidences, accuracies, n_bins=n_bins)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
    
    # Plot individual predictions as small dots
    ax.scatter(confidences, accuracies, alpha=0.3, s=8, color='lightblue', 
               label='Individual Predictions')
    
    # Calculate and plot bin averages
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_counts.append(in_bin.sum())
    
    if bin_centers:
        # Plot bin averages as large red dots
        ax.scatter(bin_centers, bin_accuracies, s=100, color='red', 
                   edgecolors='darkred', linewidth=2, zorder=5,
                   label='Bin Averages (What ECE Measures)')
        
        # Add count labels on each bin
        for x, y, count in zip(bin_centers, bin_accuracies, bin_counts):
            ax.annotate(f'{count}', (x, y), xytext=(0, 10), 
                       textcoords='offset points', ha='center', 
                       fontsize=8, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{title}\n{method_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add metrics text
    metrics_text = f'ECE = {ece:.3f}\nMCE = {mce:.3f}'
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ece, mce

def create_calibration_method_comparison():
    """Create individual reliability diagrams for each calibration method."""
    
    # Load data and simulate confidence (same as main demo)
    test_sets = get_test_sets()
    all_examples = test_sets['all']
    np.random.seed(42)
    
    confidences = []
    accuracies = []
    
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
    
    # Train calibrators
    platt_cal = PlattScalingCalibrator()
    platt_cal.fit(confidences, accuracies)
    platt_confidences = platt_cal.predict_proba(confidences)
    
    isotonic_cal = IsotonicRegressionCalibrator()
    isotonic_cal.fit(confidences, accuracies)
    isotonic_confidences = isotonic_cal.predict_proba(confidences)
    
    # Create individual charts
    methods = [
        (confidences, "Raw Model Output", "Before Calibration"),
        (platt_confidences, "Platt Scaling", "Logistic Regression Calibration"),
        (isotonic_confidences, "Isotonic Regression", "Non-parametric Calibration")
    ]
    
    for conf_data, method_name, description in methods:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        ece, mce = plot_single_reliability_diagram(ax, conf_data, accuracies, 
                                                   description, method_name)
        
        # Add interpretation text (only for calibration methods, not raw model)
        if method_name == "Raw Model Output":
            # Clean chart without text box - explanation goes in README
            plt.tight_layout()
        elif method_name == "Platt Scaling":
            # Clean chart without text box - explanation goes in README
            plt.tight_layout()
        else:  # Isotonic
            # Clean chart without text box - explanation goes in README
            plt.tight_layout()
        
        # Save individual chart
        filename = f"images/calibration/reliability_{method_name.lower().replace(' ', '_')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"💾 {method_name} reliability diagram saved to {filename}")
        print(f"   ECE: {ece:.3f}, MCE: {mce:.3f}")
    
    print("✅ All calibration method diagrams created!")

if __name__ == "__main__":
    print("📊 Creating calibration method comparison diagrams...")
    create_calibration_method_comparison()
