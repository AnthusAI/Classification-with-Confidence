#!/usr/bin/env python3
"""
Business Decision Reliability Visualization

This script creates a focused visualization showing the relationship between
confidence thresholds and actual accuracy for business decision making.

Key insight: Well-calibrated confidence means "90% confident = 90% correct"
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from datasets import get_test_sets
from calibration import IsotonicRegressionCalibrator
from calibration_metrics import calibration_metrics

def create_business_reliability_visualization():
    """Create visualization showing confidence threshold vs actual accuracy."""
    
    print("🎯 Creating Business Decision Reliability visualization...")
    
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
    
    # Create threshold analysis - FULL RANGE from 0% to 100%
    thresholds = np.arange(0.0, 1.01, 0.02)  # 0% to 100% in 2% increments
    
    raw_accuracies = []
    cal_accuracies = []
    
    for threshold in thresholds:
        # Raw confidence analysis
        raw_mask = confidences >= threshold
        if raw_mask.sum() > 0:
            raw_acc = accuracies[raw_mask].mean()
            raw_accuracies.append(raw_acc)
        else:
            raw_accuracies.append(np.nan)
        
        # Calibrated confidence analysis
        cal_mask = calibrated_confidences >= threshold
        if cal_mask.sum() > 0:
            cal_acc = accuracies[cal_mask].mean()
            cal_accuracies.append(cal_acc)
        else:
            cal_accuracies.append(np.nan)
    
    # Create the visualization
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot accuracy vs threshold
    valid_raw = ~np.isnan(raw_accuracies)
    valid_cal = ~np.isnan(cal_accuracies)
    
    ax.plot(thresholds[valid_raw] * 100, np.array(raw_accuracies)[valid_raw] * 100,
             'ro-', linewidth=3, markersize=6, label='Raw Confidence', alpha=0.8)
    ax.plot(thresholds[valid_cal] * 100, np.array(cal_accuracies)[valid_cal] * 100,
             'go-', linewidth=3, markersize=6, label='Calibrated Confidence', alpha=0.8)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=2, alpha=0.7, label='Perfect Calibration')
    
    ax.set_xlabel('Confidence Threshold (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Business Decision Reliability\nConfidence Threshold vs Actual Accuracy', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add 90% calibration crosshairs
    ax.axvline(90, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    ax.axhline(90, color='purple', linestyle='--', linewidth=2, alpha=0.8)
    
    # Clean chart without confusing text box
    
    plt.tight_layout()
    plt.savefig("images/calibration/business_reliability.png", dpi=300, bbox_inches='tight')
    print("💾 Business reliability visualization saved to images/calibration/business_reliability.png")
    plt.close()

if __name__ == "__main__":
    create_business_reliability_visualization()
