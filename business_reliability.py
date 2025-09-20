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
from dataset_loader import DatasetLoader
from calibration import IsotonicRegressionCalibrator
from calibration_metrics import calibration_metrics

def create_business_reliability_visualization():
    """Create visualization showing confidence threshold vs actual accuracy."""
    
    print("🎯 Creating Business Decision Reliability visualization...")
    
    # Load real data and simulate confidence patterns (same as calibration_demo.py)
    from dataset_loader import DatasetLoader
    loader = DatasetLoader()
    all_examples = loader.load_all()
    
    np.random.seed(42)  # Same seed for consistency
    
    confidences = []
    accuracies = []
    
    for example in all_examples:
        category = example.get('category', 'unknown')
        
        # Same simulation logic as calibration_demo.py with updated category names
        if category in ['strong_positive', 'strong_negative']:
            true_difficulty = np.random.beta(1, 9)
            confidence = 0.85 + 0.14 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.05 - 0.05 * np.random.random()
        elif category in ['medium_positive', 'medium_negative']:
            true_difficulty = np.random.beta(2, 6)
            confidence = 0.75 + 0.20 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.08 - 0.07 * np.random.random()
        elif category in ['weak_positive', 'weak_negative']:
            true_difficulty = np.random.beta(3, 3)
            confidence = 0.55 + 0.25 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.12 - 0.13 * np.random.random()
        elif category in ['neutral_positive', 'neutral_negative']:
            true_difficulty = np.random.beta(5, 2)
            confidence = 0.45 + 0.25 * (1 - true_difficulty)
            accuracy_prob = confidence - 0.20 - 0.15 * np.random.random()
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
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Add colored background regions to show business decision zones
    # High confidence zone (90-100%): Green background
    ax.axvspan(90, 100, alpha=0.15, color='green', label='High Confidence Zone (≥90%)')
    
    # Medium confidence zone (70-90%): Yellow background  
    ax.axvspan(70, 90, alpha=0.15, color='yellow', label='Medium Confidence Zone (70-90%)')
    
    # Low confidence zone (50-70%): Orange background
    ax.axvspan(50, 70, alpha=0.15, color='orange', label='Low Confidence Zone (50-70%)')
    
    # Very low confidence zone (0-50%): Red background
    ax.axvspan(0, 50, alpha=0.15, color='red', label='Very Low Confidence Zone (<50%)')
    
    # Plot accuracy vs threshold
    valid_raw = ~np.isnan(raw_accuracies)
    valid_cal = ~np.isnan(cal_accuracies)
    
    ax.plot(thresholds[valid_raw] * 100, np.array(raw_accuracies)[valid_raw] * 100,
             'ro-', linewidth=4, markersize=8, label='Raw Confidence', alpha=0.9, zorder=5)
    ax.plot(thresholds[valid_cal] * 100, np.array(cal_accuracies)[valid_cal] * 100,
             'go-', linewidth=4, markersize=8, label='Calibrated Confidence', alpha=0.9, zorder=5)
    ax.plot([0, 100], [0, 100], 'k--', linewidth=3, alpha=0.8, label='Perfect Calibration', zorder=4)
    
    # Add key business threshold lines
    ax.axvline(90, color='purple', linestyle='--', linewidth=3, alpha=0.9, zorder=3, 
               label='90% Threshold (Business Decision Point)')
    ax.axhline(90, color='purple', linestyle='--', linewidth=3, alpha=0.9, zorder=3)
    
    # Add sample count annotations for key thresholds
    key_thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    for threshold in key_thresholds:
        threshold_idx = np.argmin(np.abs(thresholds - threshold))
        if threshold_idx < len(thresholds):
            # Count samples above this threshold for calibrated model
            cal_mask = calibrated_confidences >= threshold
            sample_count = cal_mask.sum()
            
            if sample_count > 0:
                cal_acc = accuracies[cal_mask].mean()
                ax.annotate(f'{sample_count} samples\n{cal_acc:.1%} accurate', 
                           xy=(threshold * 100, cal_acc * 100),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'),
                           fontsize=10, ha='left')
    
    ax.set_xlabel('Confidence Threshold (%)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual Accuracy (%)', fontsize=16, fontweight='bold')
    ax.set_title('Business Decision Reliability: Confidence Buckets & Thresholds\n' +
                 'Each colored region shows a different confidence bucket for business decisions', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # Add business interpretation text box
    business_text = ("Business Decision Guide:\n" +
                    "• Green Zone (≥90%): Auto-approve predictions\n" +
                    "• Yellow Zone (70-90%): Human review recommended\n" +
                    "• Orange Zone (50-70%): High uncertainty, manual check\n" +
                    "• Red Zone (<50%): Very uncertain, manual processing")
    ax.text(0.02, 0.98, business_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy'),
            fontsize=11, verticalalignment='top', fontweight='bold')
    
    # Clean chart without confusing text box
    
    plt.tight_layout()
    plt.savefig("images/calibration/business_reliability.png", dpi=300, bbox_inches='tight')
    print("💾 Business reliability visualization saved to images/calibration/business_reliability.png")
    plt.close()

if __name__ == "__main__":
    create_business_reliability_visualization()
