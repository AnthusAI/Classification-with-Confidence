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

# Set consistent style for all charts
plt.style.use('seaborn-v0_8-pastel')

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
                                    title: str, filename: str, n_bins: int = 20,
                                    x_label: str = "Confidence"):
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
            bin_accuracies.append(np.mean(accuracies[in_bin]) * 100)  # Convert to percentage
            bin_confidences.append(np.mean(confidences[in_bin]) * 100)  # Convert to percentage
            bin_counts.append(np.sum(in_bin))
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2 * 100)  # Convert to percentage
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
        left = (i / n_bins) * 100
        right = ((i + 1) / n_bins) * 100
        if i % 2 == 0:  # Alternate stripe pattern
            ax.axvspan(left, right, alpha=0.12, color='lightblue', zorder=0)
    
    # Add bucket boundary lines (FULL RANGE)
    for i in range(1, n_bins):
        ax.axvline((i / n_bins) * 100, color='lightgray', linestyle=':', alpha=0.5, zorder=1)
    
    # Plot perfect calibration line
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
    
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
        
        # Plot with proportional sizes (data already converted to percentage)
        scatter = ax.scatter(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                  s=sizes, alpha=0.7, linewidth=1.5,
                  label='Model Calibration', zorder=5)
        
        # Set edge colors to darker version of face color
        face_color = scatter.get_facecolors()[0] if len(scatter.get_facecolors()) > 0 else plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
        # Make edge color darker by reducing brightness
        import matplotlib.colors as mcolors
        if hasattr(face_color, '__len__') and len(face_color) >= 3:
            darker_color = [max(0, c * 0.7) for c in face_color[:3]]  # Make 30% darker
            if len(face_color) > 3:
                darker_color.append(face_color[3])  # Keep alpha
        else:
            darker_color = 'darkblue'  # Fallback
        scatter.set_edgecolors(darker_color)
        
        # Connect dots with lines
        ax.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                '-', linewidth=2, alpha=0.6, zorder=4)
        
        # Add LARGER sample count annotations
        for conf, acc, count in zip(bin_confidences[valid_bins], 
                                   bin_accuracies[valid_bins], 
                                   bin_counts[valid_bins]):
            if count > 0:
                ax.annotate(f'{int(count)}', (conf, acc), xytext=(0, 0), 
                           textcoords='offset points', fontsize=12, fontweight='bold',
                           ha='center', va='center', color='black', zorder=6)
    
    # Convert to percentage scales (0-100% range)
    ax.set_xlim(-2, 102)  # 0-100% with padding
    ax.set_ylim(-2, 102)  # 0-100% with padding
    
    # Set percentage ticks
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Create legend with smaller marker for model calibration
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
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
    
    # Get colors from current style
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    base_color = colors[0] if len(colors) > 0 else '#1f77b4'
    ft_color = colors[1] if len(colors) > 1 else '#ff7f0e'
    
    # Convert colors to RGBA with alpha for stripes
    import matplotlib.colors as mcolors
    base_rgba = mcolors.to_rgba(base_color, alpha=0.25)
    ft_rgba = mcolors.to_rgba(ft_color, alpha=0.25)
    
    for ax, confidences, accuracies, model_name, stripe_color, dot_color in [
        (ax1, base_conf, base_acc, 'Base Model', base_rgba, base_color),
        (ax2, ft_conf, ft_acc, 'Fine-Tuned Model', ft_rgba, ft_color)
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
                bin_accuracies.append(np.mean(accuracies[in_bin]) * 100)  # Convert to percentage
                bin_confidences.append(np.mean(confidences[in_bin]) * 100)  # Convert to percentage
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_lower + bin_upper) / 2 * 100)  # Convert to percentage
                bin_counts.append(0)
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Add striped buckets FULL RANGE (0-100%)
        for i in range(n_bins):
            left = (i / n_bins) * 100
            right = ((i + 1) / n_bins) * 100
            if i % 2 == 0:  # Alternate stripe pattern
                ax.axvspan(left, right, alpha=0.12, color=stripe_color, zorder=0)
        
        # Add bucket boundary lines (FULL RANGE)
        for i in range(1, n_bins):
            ax.axvline((i / n_bins) * 100, color='lightgray', linestyle=':', alpha=0.5, zorder=1)
        
        # Plot perfect calibration line
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
        
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
            scatter = ax.scatter(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                      s=sizes, color=dot_color, alpha=0.7, linewidth=1.5,
                      label=f'{model_name} Calibration', zorder=5)
            
            # Set edge colors to darker version of face color
            import matplotlib.colors as mcolors
            try:
                # Convert color to RGB if it's a named color or hex
                rgb_color = mcolors.to_rgb(dot_color)
                darker_color = [max(0, c * 0.7) for c in rgb_color]  # Make 30% darker
                scatter.set_edgecolors([darker_color])
            except:
                scatter.set_edgecolors('darkblue')  # Fallback
            
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
                               ha='center', va='center', color='black', zorder=6)
        
        # Calculate metrics
        ece = expected_calibration_error(confidences, accuracies, n_bins)
        mce = maximum_calibration_error(confidences, accuracies, n_bins)
        
        # Convert to percentage scales (0-100% range)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        
        # Set percentage ticks
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
        
        ax.set_xlabel('Confidence', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
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

def create_confidence_histogram(confidences: List[float], title: str, filename: str, subtitle: str = None):
    """Create confidence distribution histogram with square aspect ratio and context."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create histogram with consistent scientific blue color
    n, bins, patches = ax.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Total Probability Score', fontsize=14)
    ax.set_ylabel('Number of Predictions', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Add subtitle with context if provided
    if subtitle:
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, fontsize=12, 
                ha='center', va='top', style='italic', color='#555555')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f'✅ Saved {filename}')

def create_confidence_threshold_counts(results: List[Dict], title: str, filename: str):
    """Create chart showing prediction counts above different confidence thresholds."""
    confidences = np.array([r['confidence'] for r in results])
    
    # Define thresholds (SAME as the other chart)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    counts = []
    
    for threshold in thresholds:
        count = np.sum(confidences >= threshold)
        counts.append(count)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.bar([f'{int(t*100)}%' for t in thresholds], counts, 
                  color='#2E86AB', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_xlabel('Confidence Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Predictions Above Threshold', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved {filename}')

def create_accuracy_vs_threshold_tradeoff(results: List[Dict], title: str, filename: str):
    """Create chart showing accuracy vs number of predictions trade-off."""
    confidences = np.array([r['confidence'] for r in results])
    accuracies = np.array([r['correct'] for r in results])
    
    # Define thresholds (SAME as the other chart)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    threshold_accuracies = []
    threshold_counts = []
    
    for threshold in thresholds:
        above_threshold = confidences >= threshold
        if np.sum(above_threshold) > 0:
            acc = np.mean(accuracies[above_threshold])
            count = np.sum(above_threshold)
        else:
            acc = 0
            count = 0
        threshold_accuracies.append(acc * 100)  # Convert to percentage
        threshold_counts.append(count)
    
    # Create chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Add secondary y-axis for counts (plot bars first, lower layer)
    ax2 = ax.twinx()
    bars = ax2.bar([f'{int(t*100)}%' for t in thresholds], threshold_counts, 
                   alpha=0.6, label='Prediction Count', color='C0', zorder=1)
    
    # Plot accuracy line (plot second, higher layer to appear on top)
    line = ax.plot([f'{int(t*100)}%' for t in thresholds], threshold_accuracies, 
                   'o-', linewidth=3, markersize=8, label='Accuracy', color='C2', zorder=3)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, threshold_counts)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add accuracy labels on dots
    for i, (threshold, acc) in enumerate(zip(thresholds, threshold_accuracies)):
        ax.annotate(f'{acc:.1f}%', 
                   (i, acc), 
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', va='bottom', fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Confidence Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='C2')
    ax2.set_ylabel('Number of Predictions', fontsize=14, fontweight='bold', color='C0')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Color the y-axis labels to match the data
    ax.tick_params(axis='y', labelcolor='C2', labelsize=12)
    ax2.tick_params(axis='y', labelcolor='C0', labelsize=12)
    ax.tick_params(axis='x', labelsize=10, rotation=45)
    
    ax.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight')
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

def create_sample_size_calibration_analysis(results: List[Dict], title: str, filename: str):
    """Show how calibration quality improves with more samples."""
    from sklearn.isotonic import IsotonicRegression
    
    # Test different sample sizes up to 2000
    sample_sizes = [50, 100, 200, 500, 1000, 2000]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Calculate shared Y-axis limits for consistency
    max_ece = 0
    for sample_size in sample_sizes:
        if sample_size <= len(results):
            sample_data = results[:sample_size]
            confidences = np.array([r['confidence'] for r in sample_data])
            accuracies = np.array([r['correct'] for r in sample_data])
            ece = expected_calibration_error(confidences, accuracies, 10)
            max_ece = max(max_ece, ece)
    
    y_limit = max_ece * 1.1
    
    for i, sample_size in enumerate(sample_sizes):
        ax = axes[i]
        
        if sample_size <= len(results):
            # Use first N samples
            sample_data = results[:sample_size]
            confidences = np.array([r['confidence'] for r in sample_data])
            accuracies = np.array([r['correct'] for r in sample_data])
            
            # Apply isotonic regression calibration
            iso_reg = IsotonicRegression(out_of_bounds='clip')
            calibrated_confidences = iso_reg.fit_transform(confidences, accuracies)
            
            # Calculate calibration quality
            ece_raw = expected_calibration_error(confidences, accuracies, 10)
            ece_calibrated = expected_calibration_error(calibrated_confidences, accuracies, 10)
            
            # Create reliability plot
            bin_boundaries = np.linspace(0, 1, 11)
            bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for j in range(10):
                mask = (calibrated_confidences >= bin_boundaries[j]) & (calibrated_confidences < bin_boundaries[j + 1])
                if j == 9:  # Include upper boundary for last bin
                    mask = (calibrated_confidences >= bin_boundaries[j]) & (calibrated_confidences <= bin_boundaries[j + 1])
                
                if np.sum(mask) > 0:
                    bin_accuracies.append(np.mean(accuracies[mask]))
                    bin_confidences.append(np.mean(calibrated_confidences[mask]))
                    bin_counts.append(np.sum(mask))
                else:
                    bin_accuracies.append(0)
                    bin_confidences.append(bin_centers[j])
                    bin_counts.append(0)
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1)
            
            # Plot calibration curve
            valid_bins = np.array(bin_counts) > 0
            if np.any(valid_bins):
                ax.plot(np.array(bin_confidences)[valid_bins], np.array(bin_accuracies)[valid_bins], 
                       'o-', linewidth=2, markersize=6)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Title with sample size and ECE
            ax.set_title(f'{sample_size} Samples\nECE: {ece_calibrated:.3f}', 
                        fontsize=12, fontweight='bold')
            
            if i >= 3:  # Bottom row
                ax.set_xlabel('Calibrated Confidence', fontsize=10)
            if i % 3 == 0:  # Left column
                ax.set_ylabel('Accuracy', fontsize=10)
        else:
            ax.text(0.5, 0.5, f'Need {sample_size}\nsamples', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight')
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
            if sample_size <= len(base_results):
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
            else:
                # Not enough data for this sample size
                predictions_at_threshold.append(0)
                accuracy_at_threshold.append(0)
        
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
        
        # Plot with consistent Y-axis (remove explicit colors to use seaborn style)
        bars = ax.bar(range(len(sample_sizes)), predictions_at_threshold, 
                      alpha=0.7)
        
        # Set edge colors to darker version of face colors
        import matplotlib.colors as mcolors
        for bar in bars:
            face_color = bar.get_facecolor()
            try:
                # Convert to RGB and make 30% darker
                rgb_color = mcolors.to_rgb(face_color)
                darker_color = [max(0, c * 0.7) for c in rgb_color]
                bar.set_edgecolor(darker_color)
            except:
                bar.set_edgecolor('darkblue')  # Fallback
        
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
    
    # Get colors from current style sheet
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    base_color = colors[0] if len(colors) > 0 else '#1f77b4'
    ft_color = colors[1] if len(colors) > 1 else '#ff7f0e'
    chart_colors = [base_color, ft_color]
    
    bars = ax.bar(models, accuracies, alpha=0.8, linewidth=2, color=chart_colors)
    
    # Set edge colors to darker version of face colors
    import matplotlib.colors as mcolors
    for bar in bars:
        face_color = bar.get_facecolor()
        try:
            # Convert to RGB and make 30% darker
            rgb_color = mcolors.to_rgb(face_color)
            darker_color = [max(0, c * 0.7) for c in rgb_color]
            bar.set_edgecolor(darker_color)
        except:
            bar.set_edgecolor('darkblue')  # Fallback
    
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
    
    # Get colors from current style sheet
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    base_color = colors[0] if len(colors) > 0 else '#1f77b4'
    ft_color = colors[1] if len(colors) > 1 else '#ff7f0e'
    chart_colors = [base_color, ft_color]
    
    bars = ax.bar(models, eces, alpha=0.8, linewidth=2, color=chart_colors)
    
    # Set edge colors to darker version of face colors
    import matplotlib.colors as mcolors
    for bar in bars:
        face_color = bar.get_facecolor()
        try:
            # Convert to RGB and make 30% darker
            rgb_color = mcolors.to_rgb(face_color)
            darker_color = [max(0, c * 0.7) for c in rgb_color]
            bar.set_edgecolor(darker_color)
        except:
            bar.set_edgecolor('darkblue')  # Fallback
    
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
    
    # Get colors from current style sheet
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    base_color = colors[0] if len(colors) > 0 else '#1f77b4'
    ft_color = colors[1] if len(colors) > 1 else '#ff7f0e'
    
    # Base Model (Left) - Clean, no mean clutter
    n1, bins1, patches1 = ax1.hist(base_conf, bins=50, alpha=0.8, color=base_color)
    
    # Set edge colors to darker version of face color for base model
    import matplotlib.colors as mcolors
    try:
        rgb_color = mcolors.to_rgb(base_color)
        darker_color = [max(0, c * 0.7) for c in rgb_color]
        for patch in patches1:
            patch.set_edgecolor(darker_color)
    except:
        for patch in patches1:
            patch.set_edgecolor('darkblue')
    
    ax1.set_xlabel('Confidence Score', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Predictions', fontsize=14, fontweight='bold')
    ax1.set_title('Base Model', fontsize=16, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, y_limit)
    ax1.tick_params(labelsize=12)
    
    # Fine-Tuned Model (Right) - Clean, no mean clutter
    n2, bins2, patches2 = ax2.hist(ft_conf, bins=50, alpha=0.8, color=ft_color)
    
    # Set edge colors to darker version of face color for fine-tuned model
    try:
        rgb_color = mcolors.to_rgb(ft_color)
        darker_color = [max(0, c * 0.7) for c in rgb_color]
        for patch in patches2:
            patch.set_edgecolor(darker_color)
    except:
        for patch in patches2:
            patch.set_edgecolor('darkblue')
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
        'Total Probability Distribution',
        'raw_confidence_histogram.png',
        'Aggregated probabilities from sentiment classification predictions'
    )
    
    # Individual reliability charts
    create_enhanced_reliability_chart(
        base_confidences, base_accuracies,
        'Raw Total Probability vs Actual Accuracy',
        'reliability_raw_model_output.png',
        x_label='Total Probability'
    )
    
    create_enhanced_reliability_chart(
        base_platt, base_accuracies,
        'Platt Scaling Calibration',
        'reliability_platt_scaling.png',
        x_label='Confidence (Platt Calibrated)'
    )
    
    create_enhanced_reliability_chart(
        base_isotonic, base_accuracies,
        'Isotonic Regression Calibration',
        'reliability_isotonic_regression.png',
        x_label='Confidence (Isotonic Calibrated)'
    )
    
    # Business reliability chart (dual-axis)
    create_accuracy_vs_threshold_tradeoff(
        base_results,
        'Confidence Thresholds: Accuracy vs Volume Trade-off',
        'confidence_threshold_analysis.png'
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
