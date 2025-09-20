#!/usr/bin/env python3
"""
Fast Chart Generation Using Cached Data

This script generates ALL visualization charts using pre-computed cached evaluation data.
No model loading or evaluation - just instant chart generation!

Usage:
    python generate_all_charts_fast.py

Prerequisites:
    - Run generate_evaluation_cache.py first to create cached data
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import json
import numpy as np
import os
from typing import Dict, List, Any
from calibration_metrics import expected_calibration_error, maximum_calibration_error

class CachedDataLoader:
    """Load and manage cached evaluation data."""
    
    def __init__(self):
        self.base_data = None
        self.ft_data = None
        self.metadata = None
        self._load_cache()
    
    def _load_cache(self):
        """Load all cached data."""
        print("📂 Loading cached evaluation data...")
        
        with open('evaluation_cache/base_model_results.json', 'r') as f:
            self.base_data = json.load(f)
        
        with open('evaluation_cache/finetuned_model_results.json', 'r') as f:
            self.ft_data = json.load(f)
        
        with open('evaluation_cache/evaluation_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"✅ Loaded cached data for {self.metadata['test_samples']} samples")
        print(f"   Generated: {self.metadata['generated_at']}")
    
    def get_base_results(self) -> List[Dict]:
        """Get base model evaluation results."""
        return self.base_data['results']
    
    def get_ft_results(self) -> List[Dict]:
        """Get fine-tuned model evaluation results."""
        return self.ft_data['results']
    
    def get_base_calibrated(self) -> Dict:
        """Get base model calibrated confidence scores."""
        return self.base_data['calibrated']
    
    def get_ft_calibrated(self) -> Dict:
        """Get fine-tuned model calibrated confidence scores."""
        return self.ft_data['calibrated']
    
    def get_metrics(self) -> Dict:
        """Get comprehensive metrics for both models."""
        return {
            'base': self.base_data['metrics'],
            'fine_tuned': self.ft_data['metrics']
        }

def create_enhanced_reliability_chart(confidences: np.ndarray, accuracies: np.ndarray, 
                                    title: str, filename: str, color: str = '#2E86AB', n_bins: int = 20):
    """Create enhanced reliability chart with proportional dots and full striping."""
    
    # Calculate bin statistics
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if bin_lower == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_accuracies.append(np.mean(accuracies[in_bin]))
            bin_confidences.append(np.mean(confidences[in_bin]))
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Calculate metrics
    ece = expected_calibration_error(confidences, accuracies, n_bins)
    mce = maximum_calibration_error(confidences, accuracies, n_bins)
    
    # Create square plot for proper 45-degree calibration line
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Add striped buckets FULL RANGE (0-100%)
    for i in range(n_bins):
        left = i / n_bins
        right = (i + 1) / n_bins
        if i % 2 == 0:  # Alternate stripe pattern
            ax.axvspan(left, right, alpha=0.12, color='lightblue', zorder=0)
    
    # Add bucket boundary lines (FULL RANGE)
    for i in range(1, n_bins):
        ax.axvline(i / n_bins, color='lightgray', linestyle=':', alpha=0.5, zorder=1)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
    
    # Plot reliability curve with PROPORTIONAL DOT SIZES
    valid_bins = bin_counts > 0
    if np.any(valid_bins):
        # Calculate proportional dot sizes (INCREASED for better readability)
        max_count = np.max(bin_counts[valid_bins]) if len(bin_counts[valid_bins]) > 0 else 1
        min_size, max_size = 300, 2000  # Much larger for readable labels
        sizes = []
        for count in bin_counts[valid_bins]:
            if max_count > 0:
                size = min_size + (max_size - min_size) * (count / max_count)
            else:
                size = min_size
            sizes.append(size)
        
        # Plot with proportional sizes
        ax.scatter(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                  s=sizes, color=color, alpha=0.7, edgecolors='black', linewidth=1.5,
                  label='Model Calibration', zorder=5)
        
        # Connect dots with lines
        ax.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                '-', linewidth=2, color=color, alpha=0.6, zorder=4)
        
        # Add LARGER sample count annotations
        for conf, acc, count in zip(bin_confidences[valid_bins], 
                                   bin_accuracies[valid_bins], 
                                   bin_counts[valid_bins]):
            if count > 0:
                ax.annotate(f'{int(count)}', (conf, acc), xytext=(0, 0), 
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           ha='center', va='center', color='white', zorder=6)
    
    # Set limits with small padding to prevent edge truncation
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel('Mean Predicted Confidence', fontsize=14)
    ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Create legend with smaller marker for model calibration
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=8, label='Model Calibration', markeredgecolor='black'),  # Much smaller: 8 vs 32
        plt.Line2D([0], [0], color='black', linestyle='--', label='Perfect Calibration')
    ]
    ax.legend(handles=legend_elements, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Force square aspect ratio for true 45-degree diagonal
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f'✅ Saved {filename}')

def create_side_by_side_reliability(base_conf: np.ndarray, base_acc: np.ndarray,
                                   ft_conf: np.ndarray, ft_acc: np.ndarray,
                                   title: str, filename: str, n_bins: int = 20):
    """Create side-by-side reliability comparison."""
    
    # Create side-by-side square plots for proper 45-degree calibration lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Calculate shared maximum count across both models for consistent circle scaling
    all_max_counts = []
    for confidences, accuracies in [(base_conf, base_acc), (ft_conf, ft_acc)]:
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        bin_counts = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            bin_counts.append(np.sum(in_bin))
        if len(bin_counts) > 0:
            all_max_counts.append(np.max(bin_counts))
    
    shared_max_count = max(all_max_counts) if all_max_counts else 1
    
    for ax, confidences, accuracies, model_name, stripe_color, dot_color in [
        (ax1, base_conf, base_acc, 'Base Model', 'lightblue', '#2E86AB'),
        (ax2, ft_conf, ft_acc, 'Fine-Tuned Model', 'lightgreen', '#27AE60')
    ]:
        # Calculate bin statistics
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if bin_lower == 0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_confidences.append(np.mean(confidences[in_bin]))
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Add striped buckets FULL RANGE (0-100%)
        for i in range(n_bins):
            left = i / n_bins
            right = (i + 1) / n_bins
            if i % 2 == 0:  # Alternate stripe pattern
                ax.axvspan(left, right, alpha=0.12, color=stripe_color, zorder=0)
        
        # Add bucket boundary lines (FULL RANGE)
        for i in range(1, n_bins):
            ax.axvline(i / n_bins, color='lightgray', linestyle=':', alpha=0.5, zorder=1)
        
        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
        
        # Plot reliability curve with PROPORTIONAL DOT SIZES (using shared scale)
        valid_bins = bin_counts > 0
        if np.any(valid_bins):
            # Calculate proportional dot sizes using SHARED maximum for consistent scaling
            min_size, max_size = 300, 2000  # Much larger for readable labels
            sizes = []
            for count in bin_counts[valid_bins]:
                if shared_max_count > 0:
                    size = min_size + (max_size - min_size) * (count / shared_max_count)
                else:
                    size = min_size
                sizes.append(size)
            
            # Plot with proportional sizes
            ax.scatter(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                      s=sizes, color=dot_color, alpha=0.7, edgecolors='black', linewidth=1.5,
                      label=f'{model_name} Calibration', zorder=5)
            
            # Connect dots with lines
            ax.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                    '-', linewidth=2, color=dot_color, alpha=0.6, zorder=4)
            
            # Add LARGER sample count annotations
            for conf, acc, count in zip(bin_confidences[valid_bins], 
                                       bin_accuracies[valid_bins], 
                                       bin_counts[valid_bins]):
                if count > 0:
                    ax.annotate(f'{int(count)}', (conf, acc), xytext=(0, 0), 
                               textcoords='offset points', fontsize=12, fontweight='bold',
                               ha='center', va='center', color='white', zorder=6)
        
        # Calculate metrics
        ece = expected_calibration_error(confidences, accuracies, n_bins)
        mce = maximum_calibration_error(confidences, accuracies, n_bins)
        
        # Set limits with small padding to prevent edge truncation
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Mean Predicted Confidence', fontsize=14)
        ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=14)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        
        # Create legend with smaller marker for model calibration
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=dot_color, 
                       markersize=8, label='Model Calibration', markeredgecolor='black'),  # Much smaller: 8 vs 32
            plt.Line2D([0], [0], color='black', linestyle='--', label='Perfect Calibration')
        ]
        ax.legend(handles=legend_elements, fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Force square aspect ratio for true 45-degree diagonal
        ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    os.makedirs('images/fine_tuning', exist_ok=True)
    plt.savefig(f'images/fine_tuning/{filename}', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f'✅ Saved {filename}')

def create_confidence_histogram(confidences: List[float], title: str, filename: str):
    """Create confidence distribution histogram with square aspect ratio."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create histogram with consistent scientific blue color
    n, bins, patches = ax.hist(confidences, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
    
    ax.set_xlabel('Confidence Score', fontsize=14)
    ax.set_ylabel('Number of Predictions', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Note: No aspect ratio setting for histograms - data coordinates are different scales
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f'✅ Saved {filename}')

def create_business_reliability_chart(results: List[Dict], title: str, filename: str):
    """Create business decision reliability chart."""
    confidences = [r['confidence'] for r in results]
    accuracies = [r['correct'] for r in results]
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    analysis = {}
    
    for threshold in thresholds:
        high_conf_mask = np.array(confidences) >= threshold
        high_conf_predictions = np.sum(high_conf_mask)
        
        if high_conf_predictions > 0:
            high_conf_accuracy = np.mean(np.array(accuracies)[high_conf_mask])
            false_positives = high_conf_predictions - np.sum(np.array(accuracies)[high_conf_mask])
        else:
            high_conf_accuracy = 0
            false_positives = 0
        
        analysis[threshold] = {
            'predictions': int(high_conf_predictions),
            'accuracy': high_conf_accuracy,
            'false_positives': int(false_positives)
        }
    
    # Create square business reliability chart with taller individual charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    
    # Chart 1: Predictions vs Threshold
    thresholds_list = list(analysis.keys())
    predictions = [analysis[t]['predictions'] for t in thresholds_list]
    accuracies_list = [analysis[t]['accuracy'] * 100 for t in thresholds_list]
    
    ax1.bar([str(int(t*100)) + '%' for t in thresholds_list], predictions, 
            alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Number of Predictions', fontsize=12)
    ax1.set_title('Predictions Above Threshold', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add accuracy labels on bars
    for i, (pred, acc) in enumerate(zip(predictions, accuracies_list)):
        if pred > 0:
            ax1.text(i, pred + max(predictions) * 0.02, f'{acc:.1f}%', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Chart 2: Business Impact
    ax2.plot([t*100 for t in thresholds_list], predictions, 'o-', linewidth=3, markersize=8, 
             color='#2E86AB', label='High-Confidence Predictions')
    ax2_twin = ax2.twinx()
    ax2_twin.plot([t*100 for t in thresholds_list], accuracies_list, 's-', linewidth=3, markersize=8, 
                  color='#E74C3C', label='Accuracy of Those Predictions')
    
    ax2.set_xlabel('Confidence Threshold (%)', fontsize=12)
    ax2.set_ylabel('Number of Predictions', fontsize=12, color='#2E86AB')
    ax2_twin.set_ylabel('Accuracy (%)', fontsize=12, color='#E74C3C')
    ax2.set_title('Business Decision Trade-offs', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add legends
    ax2.legend(loc='upper right')
    ax2_twin.legend(loc='center right')
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f'✅ Saved {filename}')

def create_business_reliability_progression(base_results: List[Dict], ft_results: List[Dict]):
    """Create business reliability progression chart with consistent Y-scale and square subplots."""
    sample_sizes = [50, 100, 200, 500, 1000]
    thresholds = [0.7, 0.8, 0.9, 0.95]
    
    # First pass: calculate all data to determine shared Y-axis scale
    all_predictions = []
    all_data = {}
    
    for threshold in thresholds:
        predictions_at_threshold = []
        accuracy_at_threshold = []
        
        for sample_size in sample_sizes:
            sample_results = base_results[:sample_size]
            confidences = [r['confidence'] for r in sample_results]
            accuracies = [r['correct'] for r in sample_results]
            
            high_conf_mask = np.array(confidences) >= threshold
            predictions = int(np.sum(high_conf_mask))
            
            if predictions > 0:
                accuracy = np.mean(np.array(accuracies)[high_conf_mask]) * 100
            else:
                accuracy = 0
            
            predictions_at_threshold.append(predictions)
            accuracy_at_threshold.append(accuracy)
            all_predictions.append(predictions)
        
        all_data[threshold] = (predictions_at_threshold, accuracy_at_threshold)
    
    # Calculate shared Y-axis limit
    max_predictions = max(all_predictions) if all_predictions else 1000
    y_limit = max_predictions * 1.1  # Add 10% padding
    
    # Create square subplots (taller than wide overall)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 14))
    axes = [ax1, ax2, ax3, ax4]
    
    for i, threshold in enumerate(thresholds):
        ax = axes[i]
        
        # Get pre-calculated data
        predictions_at_threshold, accuracy_at_threshold = all_data[threshold]
        
        # Plot with consistent Y-axis
        bars = ax.bar(range(len(sample_sizes)), predictions_at_threshold, 
                      alpha=0.7, color='lightblue', edgecolor='black')
        
        # Add accuracy labels
        for j, (pred, acc) in enumerate(zip(predictions_at_threshold, accuracy_at_threshold)):
            if pred > 0:
                ax.text(j, pred + y_limit * 0.02, f'{acc:.1f}%', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Sample Size', fontsize=12)
        ax.set_ylabel('High-Confidence Predictions', fontsize=12)
        ax.set_title(f'Threshold ≥{threshold*100:.0f}%', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(sample_sizes)))
        ax.set_xticklabels([str(s) for s in sample_sizes])
        ax.set_ylim(0, y_limit)  # Apply consistent Y-axis limit
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sample Size Impact on Business Decisions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig('images/calibration/business_reliability_progression.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved business_reliability_progression.png')

def create_accuracy_comparison_chart(base_results: List[Dict], ft_results: List[Dict]):
    """Create focused accuracy comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    base_acc = np.array([r['correct'] for r in base_results])
    ft_acc = np.array([r['correct'] for r in ft_results])
    
    # Calculate metrics
    base_acc_mean = np.mean(base_acc) * 100
    ft_acc_mean = np.mean(ft_acc) * 100
    
    # Accuracy Comparison
    models = ['Base Model', 'Fine-Tuned Model']
    accuracies = [base_acc_mean, ft_acc_mean]
    colors = ['#E74C3C', '#27AE60']
    
    bars = ax.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    os.makedirs('images/fine_tuning', exist_ok=True)
    plt.savefig('images/fine_tuning/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved accuracy_comparison.png')

def create_calibration_error_comparison_chart(base_results: List[Dict], ft_results: List[Dict]):
    """Create focused calibration error comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    base_conf = np.array([r['confidence'] for r in base_results])
    base_acc = np.array([r['correct'] for r in base_results])
    ft_conf = np.array([r['confidence'] for r in ft_results])
    ft_acc = np.array([r['correct'] for r in ft_results])
    
    # Calculate metrics
    base_ece = expected_calibration_error(base_conf, base_acc, 20)
    ft_ece = expected_calibration_error(ft_conf, ft_acc, 20)
    
    # Calibration Error Comparison
    models = ['Base Model', 'Fine-Tuned Model']
    eces = [base_ece, ft_ece]
    colors = ['#E74C3C', '#27AE60']
    
    bars = ax.bar(models, eces, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax.set_ylabel('Expected Calibration Error', fontsize=14, fontweight='bold')
    ax.set_title('Calibration Error Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=12)
    
    # Add value labels
    for bar, ece in zip(bars, eces):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                f'{ece:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    os.makedirs('images/fine_tuning', exist_ok=True)
    plt.savefig('images/fine_tuning/calibration_error_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved calibration_error_comparison.png')

def create_confidence_distribution_changes(base_results: List[Dict], ft_results: List[Dict]):
    """Create confidence distribution changes chart with consistent Y-axis scale and portrait orientation."""
    base_conf = [r['confidence'] for r in base_results]
    ft_conf = [r['confidence'] for r in ft_results]
    
    # Create more square figure with taller individual charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    
    # Calculate shared Y-axis limit for direct comparison
    base_counts, _ = np.histogram(base_conf, bins=50)
    ft_counts, _ = np.histogram(ft_conf, bins=50)
    max_count = max(np.max(base_counts), np.max(ft_counts))
    y_limit = max_count * 1.1  # Add 10% padding
    
    # Base Model (Left) - Clean, no mean clutter
    ax1.hist(base_conf, bins=50, alpha=0.8, color='#E74C3C', edgecolor='black')
    ax1.set_xlabel('Confidence Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Predictions', fontsize=14, fontweight='bold')
    ax1.set_title('Base Model', fontsize=16, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, y_limit)
    ax1.tick_params(labelsize=12)
    
    # Fine-Tuned Model (Right) - Clean, no mean clutter
    ax2.hist(ft_conf, bins=50, alpha=0.8, color='#27AE60', edgecolor='black')
    ax2.set_xlabel('Confidence Score', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Predictions', fontsize=14, fontweight='bold')
    ax2.set_title('Fine-Tuned Model', fontsize=16, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, y_limit)
    ax2.tick_params(labelsize=12)
    
    plt.suptitle('Confidence Distribution Changes After Fine-Tuning', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs('images/fine_tuning', exist_ok=True)
    plt.savefig('images/fine_tuning/confidence_distribution_changes.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved confidence_distribution_changes.png')

def main():
    """Generate all charts using cached data."""
    print("⚡ Fast Chart Generation Using Cached Data")
    print("=" * 50)
    
    # Load cached data
    loader = CachedDataLoader()
    
    base_results = loader.get_base_results()
    ft_results = loader.get_ft_results()
    base_calibrated = loader.get_base_calibrated()
    ft_calibrated = loader.get_ft_calibrated()
    
    # Extract arrays for plotting
    base_confidences = np.array([r['confidence'] for r in base_results])
    base_accuracies = np.array([r['correct'] for r in base_results])
    ft_confidences = np.array([r['confidence'] for r in ft_results])
    ft_accuracies = np.array([r['correct'] for r in ft_results])
    
    base_platt = np.array(base_calibrated['platt_calibrated'])
    base_isotonic = np.array(base_calibrated['isotonic_calibrated'])
    ft_platt = np.array(ft_calibrated['platt_calibrated'])
    ft_isotonic = np.array(ft_calibrated['isotonic_calibrated'])
    
    print("🎨 Generating all charts...")
    
    # 1. CALIBRATION SECTION CHARTS
    print("\\n📊 Calibration Section Charts:")
    
    # Raw confidence histograms
    create_confidence_histogram(
        base_confidences.tolist(),
        'Base Model Confidence Distribution',
        'raw_confidence_histogram.png'
    )
    
    # Individual reliability charts
    create_enhanced_reliability_chart(
        base_confidences, base_accuracies,
        'Base Model Reliability (Raw Output)',
        'reliability_raw_model_output.png'
    )
    
    create_enhanced_reliability_chart(
        base_platt, base_accuracies,
        'Base Model with Platt Scaling',
        'reliability_platt_scaling.png'
    )
    
    create_enhanced_reliability_chart(
        base_isotonic, base_accuracies,
        'Base Model with Isotonic Regression',
        'reliability_isotonic_regression.png'
    )
    
    # Business reliability charts
    create_business_reliability_chart(
        base_results,
        'Business Decision Reliability',
        'business_reliability.png'
    )
    
    create_business_reliability_progression(base_results, ft_results)
    
    # 2. FINE-TUNING SECTION CHARTS
    print("\\n🔄 Fine-Tuning Section Charts:")
    
    # Individual focused comparisons
    create_accuracy_comparison_chart(base_results, ft_results)
    create_calibration_error_comparison_chart(base_results, ft_results)
    
    # Confidence distribution changes
    create_confidence_distribution_changes(base_results, ft_results)
    
    # Side-by-side reliability comparisons
    create_side_by_side_reliability(
        base_confidences, base_accuracies,
        ft_confidences, ft_accuracies,
        'Raw Model Reliability: Base vs Fine-Tuned',
        'finetuning_raw_reliability_comparison.png'
    )
    
    create_side_by_side_reliability(
        base_platt, base_accuracies,
        ft_platt, ft_accuracies,
        'Platt Scaling Reliability: Base vs Fine-Tuned',
        'finetuning_platt_reliability_comparison.png'
    )
    
    create_side_by_side_reliability(
        base_isotonic, base_accuracies,
        ft_isotonic, ft_accuracies,
        'Isotonic Regression Reliability: Base vs Fine-Tuned',
        'finetuning_isotonic_reliability_comparison.png'
    )
    
    print("\\n🎉 ALL CHARTS generated successfully!")
    print("⚡ Total time: seconds (not minutes!)")
    print("📁 Charts saved to:")
    print("  • images/calibration/ - Individual calibration method charts + business analysis")
    print("  • images/fine_tuning/ - Side-by-side comparison charts + comprehensive analysis")

if __name__ == "__main__":
    main()
