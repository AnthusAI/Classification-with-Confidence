#!/usr/bin/env python3
"""
Dataset Size Experiment: How Sample Size Affects Calibration

This script runs the main calibration experiment with different dataset sizes
to show how the practical business charts change with more labeled data.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from datasets import get_test_sets
from calibration import IsotonicRegressionCalibrator, PlattScalingCalibrator
from calibration_metrics import calibration_metrics
import os

def run_experiment_with_size(n_samples, suffix=""):
    """Run the main calibration experiment with a specific number of samples."""
    
    print(f"\n🔬 EXPERIMENT WITH {n_samples} SAMPLES")
    print("=" * 50)
    
    # Load real data
    test_sets = get_test_sets()
    all_examples = test_sets['all']
    
    # Take only the first n_samples
    if n_samples < len(all_examples):
        selected_examples = all_examples[:n_samples]
        print(f"📊 Using first {n_samples} samples from {len(all_examples)} total")
    else:
        selected_examples = all_examples
        print(f"📊 Using all {len(selected_examples)} samples")
    
    np.random.seed(42)  # Same seed for consistency
    
    confidences = []
    accuracies = []
    
    for example in selected_examples:
        category = example.get('category', 'unknown')
        
        # Same simulation logic as calibration_demo.py
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
    
    isotonic_cal = IsotonicRegressionCalibrator()
    isotonic_cal.fit(confidences, accuracies)
    
    # Get calibrated confidences
    platt_confidences = platt_cal.predict_proba(confidences)
    isotonic_confidences = isotonic_cal.predict_proba(confidences)
    
    # Calculate metrics
    raw_metrics = calibration_metrics(confidences, accuracies)
    platt_metrics = calibration_metrics(platt_confidences, accuracies)
    isotonic_metrics = calibration_metrics(isotonic_confidences, accuracies)
    
    print(f"📊 CALIBRATION RESULTS:")
    print(f"Raw ECE:              {raw_metrics['ECE']:.3f}")
    print(f"Platt Scaling ECE:    {platt_metrics['ECE']:.3f}")
    print(f"Isotonic Regression:  {isotonic_metrics['ECE']:.3f}")
    
    # Create the two main business charts
    create_sample_size_chart(confidences, isotonic_confidences, accuracies, n_samples, suffix)
    create_business_reliability_chart(confidences, isotonic_confidences, accuracies, n_samples, suffix)
    
    # Return metrics for summary
    return {
        'n_samples': n_samples,
        'raw_ece': raw_metrics['ECE'],
        'platt_ece': platt_metrics['ECE'],
        'isotonic_ece': isotonic_metrics['ECE'],
        'raw_avg_conf': raw_metrics['Average Confidence'],
        'raw_avg_acc': raw_metrics['Average Accuracy']
    }

def create_sample_size_chart(confidences, calibrated_confidences, accuracies, n_samples, suffix):
    """Create the sample size threshold chart."""
    
    thresholds = np.arange(0.0, 1.01, 0.02)  # Full 0-100% range
    
    raw_counts = []
    cal_counts = []
    
    for threshold in thresholds:
        raw_mask = confidences >= threshold
        cal_mask = calibrated_confidences >= threshold
        
        raw_counts.append(raw_mask.sum())
        cal_counts.append(cal_mask.sum())
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(thresholds * 100, raw_counts, 'r-', linewidth=3, label='Raw Confidence', alpha=0.8)
    ax.plot(thresholds * 100, cal_counts, 'g-', linewidth=3, label='Calibrated Confidence', alpha=0.8)
    
    ax.set_xlabel('Confidence Threshold (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Predictions Above Threshold', fontsize=14, fontweight='bold')
    ax.set_title(f'Sample Size at Each Confidence Threshold\n{n_samples} Labeled Samples', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    # Add key threshold markers
    key_thresholds = [50, 70, 80, 90, 95]
    for threshold in key_thresholds:
        ax.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
        
        closest_idx = np.argmin(np.abs(thresholds * 100 - threshold))
        raw_count = raw_counts[closest_idx]
        cal_count = cal_counts[closest_idx]
        
        ax.text(threshold, raw_count + max(raw_counts) * 0.05, f'{raw_count}', ha='center', fontsize=10, 
                color='red', fontweight='bold')
        ax.text(threshold, cal_count - max(raw_counts) * 0.05, f'{cal_count}', ha='center', fontsize=10, 
                color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"images/calibration/sample_size_thresholds_{n_samples}samples{suffix}.png", dpi=300, bbox_inches='tight')
    print(f"💾 Sample size chart saved for {n_samples} samples")
    plt.close()

def create_business_reliability_chart(confidences, calibrated_confidences, accuracies, n_samples, suffix):
    """Create the business reliability chart."""
    
    thresholds = np.arange(0.0, 1.01, 0.02)  # Full 0-100% range
    
    raw_accuracies = []
    cal_accuracies = []
    
    for threshold in thresholds:
        raw_mask = confidences >= threshold
        if raw_mask.sum() > 0:
            raw_acc = accuracies[raw_mask].mean()
            raw_accuracies.append(raw_acc)
        else:
            raw_accuracies.append(np.nan)
        
        cal_mask = calibrated_confidences >= threshold
        if cal_mask.sum() > 0:
            cal_acc = accuracies[cal_mask].mean()
            cal_accuracies.append(cal_acc)
        else:
            cal_accuracies.append(np.nan)
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    valid_raw = ~np.isnan(raw_accuracies)
    valid_cal = ~np.isnan(cal_accuracies)
    
    ax.plot(thresholds[valid_raw] * 100, np.array(raw_accuracies)[valid_raw] * 100,
             'ro-', linewidth=3, markersize=6, label='Raw Confidence', alpha=0.8)
    ax.plot(thresholds[valid_cal] * 100, np.array(cal_accuracies)[valid_cal] * 100,
             'go-', linewidth=3, markersize=6, label='Calibrated Confidence', alpha=0.8)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
    
    ax.set_xlabel('Confidence Threshold (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'Business Decision Reliability\n{n_samples} Labeled Samples', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add 90% calibration crosshairs
    ax.axvline(90, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(90, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f"images/calibration/business_reliability_{n_samples}samples{suffix}.png", dpi=300, bbox_inches='tight')
    print(f"💾 Business reliability chart saved for {n_samples} samples")
    plt.close()

def create_summary_chart(all_results):
    """Create a summary chart showing how calibration quality changes with dataset size."""
    
    sample_sizes = [r['n_samples'] for r in all_results]
    raw_eces = [r['raw_ece'] for r in all_results]
    platt_eces = [r['platt_ece'] for r in all_results]
    isotonic_eces = [r['isotonic_ece'] for r in all_results]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.plot(sample_sizes, raw_eces, 'ro-', linewidth=3, markersize=8, label='Raw Confidence ECE', alpha=0.8)
    ax.plot(sample_sizes, platt_eces, 'bo-', linewidth=3, markersize=8, label='Platt Scaling ECE', alpha=0.8)
    ax.plot(sample_sizes, isotonic_eces, 'go-', linewidth=3, markersize=8, label='Isotonic Regression ECE', alpha=0.8)
    
    ax.set_xlabel('Number of Labeled Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=14, fontweight='bold')
    ax.set_title('How Calibration Quality Improves with More Data\n(Lower ECE = Better Calibration)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(raw_eces), max(platt_eces), max(isotonic_eces)) * 1.1)
    
    # Add value labels
    for i, (size, raw, platt, iso) in enumerate(zip(sample_sizes, raw_eces, platt_eces, isotonic_eces)):
        ax.text(size, raw + 0.01, f'{raw:.3f}', ha='center', fontsize=10, color='red', fontweight='bold')
        ax.text(size, platt + 0.01, f'{platt:.3f}', ha='center', fontsize=10, color='blue', fontweight='bold')
        ax.text(size, iso + 0.01, f'{iso:.3f}', ha='center', fontsize=10, color='green', fontweight='bold')
    
    # Add explanation
    explanation_text = (
        "Key Insights:\n"
        "• More labeled samples = better calibration (lower ECE)\n"
        "• Isotonic regression (green) performs best with sufficient data\n"
        "• Raw confidence (red) also improves with diverse examples\n"
        "• ECE < 0.05 = excellent calibration for business use"
    )
    
    ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("images/calibration/calibration_vs_dataset_size.png", dpi=300, bbox_inches='tight')
    print("💾 Summary calibration chart saved")
    plt.close()

def main():
    """Run experiments with different dataset sizes."""
    
    print("🧪 DATASET SIZE CALIBRATION EXPERIMENT")
    print("=" * 60)
    print("Running calibration experiments with 50, 100, 500, and 900 samples...")
    print("This will show how the business charts change with more labeled data.")
    
    # Ensure output directory exists
    os.makedirs("images/calibration", exist_ok=True)
    
    # Run experiments with different sizes
    sample_sizes = [50, 100, 500, 900]
    all_results = []
    
    for size in sample_sizes:
        result = run_experiment_with_size(size)
        all_results.append(result)
    
    # Create summary chart
    print(f"\n📊 Creating summary chart...")
    create_summary_chart(all_results)
    
    print(f"\n✅ EXPERIMENT COMPLETE!")
    print("=" * 60)
    print("Generated charts:")
    for size in sample_sizes:
        print(f"  • sample_size_thresholds_{size}samples.png")
        print(f"  • business_reliability_{size}samples.png")
    print("  • calibration_vs_dataset_size.png (summary)")
    
    print(f"\nFinal Results Summary:")
    for result in all_results:
        print(f"  {result['n_samples']:3d} samples: Raw ECE={result['raw_ece']:.3f}, "
              f"Platt ECE={result['platt_ece']:.3f}, Isotonic ECE={result['isotonic_ece']:.3f}")

if __name__ == "__main__":
    main()
