#!/usr/bin/env python3
"""
Sample Size at Confidence Thresholds Visualization

This script creates a focused visualization showing how many predictions
are available at different confidence thresholds for both raw and calibrated
confidence scores.

Key insight: Higher thresholds = fewer predictions but potentially more reliable
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from datasets import get_test_sets
from calibration import IsotonicRegressionCalibrator
from calibration_metrics import calibration_metrics

def create_sample_size_visualization():
    """Create visualization showing sample counts at different confidence thresholds."""
    
    print("📊 Creating Sample Size at Thresholds visualization...")
    
    # Load real data and simulate confidence patterns (same as calibration_demo.py)
    test_sets = get_test_sets()
    all_examples = test_sets['all']
    
    np.random.seed(42)  # Same seed for consistency
    
    confidences = []
    accuracies = []
    
    for example in all_examples:
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
    
    # Train isotonic calibrator
    isotonic_cal = IsotonicRegressionCalibrator()
    isotonic_cal.fit(confidences, accuracies)
    calibrated_confidences = isotonic_cal.predict_proba(confidences)
    
    # Create threshold analysis
    thresholds = np.arange(0.3, 1.0, 0.02)
    
    raw_counts = []
    cal_counts = []
    
    for threshold in thresholds:
        raw_mask = confidences >= threshold
        cal_mask = calibrated_confidences >= threshold
        
        raw_counts.append(raw_mask.sum())
        cal_counts.append(cal_mask.sum())
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot sample counts
    ax.plot(thresholds * 100, raw_counts, 'r-', linewidth=3, label='Raw Confidence', alpha=0.8)
    ax.plot(thresholds * 100, cal_counts, 'g-', linewidth=3, label='Calibrated Confidence', alpha=0.8)
    
    ax.set_xlabel('Confidence Threshold (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Predictions Above Threshold', fontsize=14, fontweight='bold')
    ax.set_title('Sample Size at Each Confidence Threshold\nHow Many Predictions Are Available for Decision Making?', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(30, 100)
    
    # Add key threshold markers
    key_thresholds = [50, 70, 80, 90, 95]
    for threshold in key_thresholds:
        ax.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
        
        # Find closest threshold index
        closest_idx = np.argmin(np.abs(thresholds * 100 - threshold))
        raw_count = raw_counts[closest_idx]
        cal_count = cal_counts[closest_idx]
        
        # Add count labels
        ax.text(threshold, raw_count + 20, f'{raw_count}', ha='center', fontsize=10, 
                color='red', fontweight='bold')
        ax.text(threshold, cal_count - 30, f'{cal_count}', ha='center', fontsize=10, 
                color='green', fontweight='bold')
    
    # Add explanation text
    explanation_text = (
        "Key Insights:\n"
        "• Higher thresholds = fewer predictions available\n"
        "• Calibrated confidence is more conservative (fewer high-confidence predictions)\n"
        "• Raw confidence maintains more predictions at high thresholds\n"
        "• Trade-off: More samples vs. higher quality predictions"
    )
    
    ax.text(0.02, 0.98, explanation_text, transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig("images/calibration/sample_size_thresholds.png", dpi=300, bbox_inches='tight')
    print("💾 Sample size visualization saved to images/calibration/sample_size_thresholds.png")
    plt.close()

if __name__ == "__main__":
    create_sample_size_visualization()
