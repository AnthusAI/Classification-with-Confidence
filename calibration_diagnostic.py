#!/usr/bin/env python3
"""
Diagnostic script to understand the disconnect between ECE metrics and decision threshold visualization.
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
from datasets import get_test_sets
from calibration import PlattScalingCalibrator, IsotonicRegressionCalibrator
from calibration_metrics import expected_calibration_error

def analyze_calibration_disconnect():
    """Analyze why ECE shows perfect calibration but decision thresholds don't."""
    
    print("🔍 CALIBRATION DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Load real data and simulate confidence patterns (same as calibration_demo.py)
    test_sets = get_test_sets()
    all_examples = test_sets['all']
    
    np.random.seed(42)  # Same seed as calibration_demo
    
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
    
    # Train calibrators
    platt_cal = PlattScalingCalibrator()
    platt_cal.fit(confidences, accuracies)
    
    isotonic_cal = IsotonicRegressionCalibrator()
    isotonic_cal.fit(confidences, accuracies)
    
    calibrated_confidences = platt_cal.predict_proba(confidences)
    
    # Calculate ECE metrics
    raw_ece = expected_calibration_error(confidences, accuracies, n_bins=10)
    cal_ece = expected_calibration_error(calibrated_confidences, accuracies, n_bins=10)
    
    print(f"📊 ECE METRICS:")
    print(f"Raw ECE: {raw_ece:.3f}")
    print(f"Calibrated ECE: {cal_ece:.3f}")
    print()
    
    # Now analyze what ECE is actually measuring vs decision thresholds
    print("🔍 ECE BINNING ANALYSIS:")
    print("-" * 40)
    
    # ECE bins (what the metric uses)
    bin_boundaries = np.linspace(0, 1, 11)  # 10 bins
    for i in range(10):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Raw confidence in this bin
        raw_mask = (confidences > bin_lower) & (confidences <= bin_upper)
        if raw_mask.sum() > 0:
            raw_avg_conf = confidences[raw_mask].mean()
            raw_avg_acc = accuracies[raw_mask].mean()
            raw_count = raw_mask.sum()
            
            print(f"Bin {bin_lower:.1f}-{bin_upper:.1f}: {raw_count:3d} samples, "
                  f"avg_conf={raw_avg_conf:.3f}, avg_acc={raw_avg_acc:.3f}, "
                  f"diff={abs(raw_avg_conf - raw_avg_acc):.3f}")
    
    print()
    print("🎯 DECISION THRESHOLD ANALYSIS:")
    print("-" * 40)
    
    # Decision thresholds (what the visualization uses)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        raw_mask = confidences >= threshold
        cal_mask = calibrated_confidences >= threshold
        
        if raw_mask.sum() > 0:
            raw_acc = accuracies[raw_mask].mean()
            raw_count = raw_mask.sum()
        else:
            raw_acc = 0
            raw_count = 0
            
        if cal_mask.sum() > 0:
            cal_acc = accuracies[cal_mask].mean()
            cal_count = cal_mask.sum()
        else:
            cal_acc = 0
            cal_count = 0
            
        print(f"Threshold {threshold:.1f}: Raw={raw_acc:.3f} ({raw_count:3d} samples), "
              f"Cal={cal_acc:.3f} ({cal_count:3d} samples)")
    
    print()
    print("💡 THE DISCONNECT:")
    print("-" * 40)
    print("• ECE measures calibration within FIXED BINS (0.0-0.1, 0.1-0.2, etc.)")
    print("• Decision thresholds measure accuracy ABOVE THRESHOLDS (>0.5, >0.7, etc.)")
    print("• These are DIFFERENT ways of slicing the data!")
    print()
    print("• ECE can be 0.000 if each bin is well-calibrated on average")
    print("• But decision thresholds can still be poor if the distribution is wrong")
    print()
    print("🔧 POTENTIAL SOLUTIONS:")
    print("1. Use different calibration methods (Temperature Scaling, Histogram Binning)")
    print("2. Optimize for decision threshold performance, not just ECE")
    print("3. Use threshold-based calibration metrics instead of bin-based ECE")

if __name__ == "__main__":
    analyze_calibration_disconnect()
