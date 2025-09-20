"""
Calibration Evaluation Metrics and Visualization

This module provides tools to measure and visualize how well-calibrated
your confidence scores are. Well-calibrated means "when the model says
it's X% confident, it should be correct X% of the time."
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings


def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, 
                             n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE).
    
    ECE measures the average difference between confidence and accuracy
    across different confidence bins. Lower is better (0 = perfect calibration).
    
    Args:
        confidences: Array of confidence scores [0, 1]
        accuracies: Array of binary correctness [0, 1]
        n_bins: Number of confidence bins to use
        
    Returns:
        ECE score (lower is better, 0 = perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Average confidence and accuracy in this bin
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add weighted calibration error for this bin
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def maximum_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, 
                            n_bins: int = 10) -> float:
    """
    Calculate Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between confidence and accuracy
    across all confidence bins. Shows worst-case calibration error.
    
    Args:
        confidences: Array of confidence scores [0, 1]
        accuracies: Array of binary correctness [0, 1]
        n_bins: Number of confidence bins to use
        
    Returns:
        MCE score (lower is better, 0 = perfect calibration)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    max_error = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this confidence bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            # Average confidence and accuracy in this bin
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Track maximum calibration error
            error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            max_error = max(max_error, error)
    
    return max_error


def brier_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """
    Calculate Brier Score.
    
    Brier score measures both calibration and sharpness (how confident
    the model is). Lower is better.
    
    Args:
        confidences: Array of confidence scores [0, 1]
        accuracies: Array of binary correctness [0, 1]
        
    Returns:
        Brier score (lower is better)
    """
    return np.mean((confidences - accuracies) ** 2)


def reliability_score(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """
    Calculate reliability (average absolute calibration error).
    
    Simple measure of how close confidence scores are to actual accuracy.
    
    Args:
        confidences: Array of confidence scores [0, 1]
        accuracies: Array of binary correctness [0, 1]
        
    Returns:
        Reliability score (lower is better, 0 = perfect)
    """
    return np.mean(np.abs(confidences - accuracies))


def calibration_metrics(confidences: np.ndarray, accuracies: np.ndarray, 
                       n_bins: int = 10) -> Dict[str, float]:
    """
    Calculate comprehensive calibration metrics.
    
    Args:
        confidences: Array of confidence scores [0, 1]
        accuracies: Array of binary correctness [0, 1]
        n_bins: Number of bins for ECE/MCE calculation
        
    Returns:
        Dictionary with all calibration metrics
    """
    return {
        'ECE': expected_calibration_error(confidences, accuracies, n_bins),
        'MCE': maximum_calibration_error(confidences, accuracies, n_bins),
        'Brier Score': brier_score(confidences, accuracies),
        'Reliability': reliability_score(confidences, accuracies),
        'Average Confidence': np.mean(confidences),
        'Average Accuracy': np.mean(accuracies)
    }


def plot_reliability_diagram(confidences: np.ndarray, accuracies: np.ndarray, 
                           n_bins: int = 10, title: str = "Reliability Diagram",
                           save_path: Optional[str] = None, show: bool = False,
                           method_name: str = "", sample_size: Optional[int] = None,
                           ax: Optional[plt.Axes] = None, show_ece: bool = False) -> plt.Figure:
    """
    Plot reliability diagram showing calibration quality.
    
    Perfect calibration appears as points on the diagonal line.
    Points above the line = overconfident, below = underconfident.
    
    Args:
        confidences: Array of confidence scores [0, 1]
        accuracies: Array of binary correctness [0, 1]
        n_bins: Number of confidence bins
        title: Plot title
        save_path: Optional path to save the plot
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_confidences.append(confidences[in_bin].mean())
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_counts.append(in_bin.sum())
    
    # Create the plot with better styling
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    else:
        fig = ax.figure
    
    # Show actual confidence buckets used in the calculation
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Create a list to track which bins have data
    bins_with_data = set()
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bins_with_data.add((bin_lower, bin_upper))
    
    # Draw bucket regions - different colors for populated vs empty buckets
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        has_data = (bin_lower, bin_upper) in bins_with_data
        
        if has_data:
            # Populated bucket - light blue background
            color = 'lightblue'
            alpha = 0.15
            edge_color = 'blue'
            edge_alpha = 0.3
            label = f'Bucket {i+1}: [{bin_lower:.1f}-{bin_upper:.1f}] (Has Data)' if i == 0 else ""
        else:
            # Empty bucket - light gray background  
            color = 'lightgray'
            alpha = 0.1
            edge_color = 'gray'
            edge_alpha = 0.2
            label = f'Empty Bucket: [{bin_lower:.1f}-{bin_upper:.1f}] (No Data)' if i == 0 and not bins_with_data else ""
        
        # Draw the bucket region as a rectangle
        rect = plt.Rectangle((bin_lower, bin_lower), bin_upper - bin_lower, bin_upper - bin_lower,
                           facecolor=color, alpha=alpha, edgecolor=edge_color, 
                           linewidth=1, linestyle='--')
        ax.add_patch(rect)
        
        # Add bucket boundary lines
        ax.axvline(bin_lower, color=edge_color, linestyle=':', alpha=edge_alpha, linewidth=1)
        ax.axhline(bin_lower, color=edge_color, linestyle=':', alpha=edge_alpha, linewidth=1)
        
        # Label the bucket in the center if it has data
        if has_data:
            center_x = (bin_lower + bin_upper) / 2
            center_y = (bin_lower + bin_upper) / 2
            ax.text(center_x, center_y, f'Bin {i+1}', ha='center', va='center', 
                   fontsize=8, alpha=0.6, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Add final boundary lines
    ax.axvline(1.0, color='blue', linestyle=':', alpha=0.3, linewidth=1)
    ax.axhline(1.0, color='blue', linestyle=':', alpha=0.3, linewidth=1)
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.8, linewidth=3, zorder=4)
    
    # Reliability points (size proportional to number of samples)
    if bin_counts:
        # Color points based on how far they are from perfect calibration
        distances = [abs(conf - acc) for conf, acc in zip(bin_confidences, bin_accuracies)]
        max_distance = max(distances) if distances else 1
        colors = ['red' if d > 0.1 else 'orange' if d > 0.05 else 'green' for d in distances]
        
        scatter = ax.scatter(bin_confidences, bin_accuracies, 
                           s=[count/max(bin_counts)*400 + 150 for count in bin_counts], 
                           alpha=0.9, c=colors, edgecolors='black', linewidth=2, zorder=5)
        
        # Add sample count labels on each point with better styling
        for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
            ax.annotate(f'{count} samples\nBin: {conf:.2f}→{acc:.2f}', (conf, acc), 
                       xytext=(8, 8), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'),
                       fontsize=9, fontweight='bold', ha='left')
    
    # Enhanced labels and title
    ax.set_xlabel('Mean Predicted Confidence', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Actual Accuracy', fontsize=14, fontweight='bold')
    
    # More informative title
    full_title = title
    if method_name:
        full_title += f" - {method_name}"
    if sample_size:
        full_title += f" (n={sample_size})"
    ax.set_title(full_title, fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Add comprehensive calibration info
    ece = expected_calibration_error(confidences, accuracies, n_bins)
    mce = maximum_calibration_error(confidences, accuracies, n_bins)
    
    # Interpretation text
    if ece < 0.05:
        interpretation = "Excellent"
        color = 'green'
    elif ece < 0.10:
        interpretation = "Good"
        color = 'orange'
    else:
        interpretation = "Poor"
        color = 'red'
    
    info_text = f'ECE: {ece:.3f} ({interpretation})\nMCE: {mce:.3f}\nSamples: {len(confidences)}'
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=2),
            fontsize=12, verticalalignment='top')
    
    # Add explanation text with bucket information
    explanation = ("CONFIDENCE BUCKETS EXPLAINED:\n"
                  "• Blue rectangles = buckets WITH data\n"
                  "• Gray rectangles = buckets WITHOUT data (empty)\n"
                  "• Dots show actual performance in populated buckets\n"
                  "• Dot size = number of predictions in that bucket\n"
                  "• Green dots = well calibrated, Red = poorly calibrated\n"
                  "• Perfect calibration = dots on diagonal line\n"
                  "• Dotted lines show bucket boundaries")
    ax.text(0.98, 0.02, explanation, transform=ax.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9, edgecolor='navy'),
            fontsize=10, horizontalalignment='right', verticalalignment='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        # Only show if we're in an interactive environment
        try:
            plt.show()
        except:
            pass  # Ignore errors in headless environments
    
    return fig


def plot_calibration_comparison(raw_confidences: np.ndarray, raw_accuracies: np.ndarray,
                               calibrated_confidences: np.ndarray, calibrated_accuracies: np.ndarray,
                               method_name: str = "Calibration Method", 
                               calibrator_name: str = "Calibrated",
                               save_path: Optional[str] = None, show: bool = False) -> plt.Figure:
    """
    Plot side-by-side comparison of before and after calibration.
    
    Args:
        raw_confidences: Original confidence scores [0, 1]
        raw_accuracies: Binary correctness for raw scores [0, 1]
        calibrated_confidences: Calibrated confidence scores [0, 1]
        calibrated_accuracies: Binary correctness for calibrated scores [0, 1]
        method_name: Name of the confidence method (e.g., "Logprobs")
        calibrator_name: Name of calibration method (e.g., "Platt Scaling")
        save_path: Optional path to save the plot
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot raw (before) calibration
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Raw calibration plot
    bin_confidences_raw = []
    bin_accuracies_raw = []
    bin_counts_raw = []
    
    # Track which bins have data for raw model
    raw_bins_with_data = set()
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (raw_confidences > bin_lower) & (raw_confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_confidences_raw.append(raw_confidences[in_bin].mean())
            bin_accuracies_raw.append(raw_accuracies[in_bin].mean())
            bin_counts_raw.append(in_bin.sum())
            raw_bins_with_data.add((bin_lower, bin_upper))
    
    # Draw bucket regions for raw model (ax1)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        has_data = (bin_lower, bin_upper) in raw_bins_with_data
        
        if has_data:
            color, alpha, edge_color, edge_alpha = 'lightcoral', 0.15, 'red', 0.3
        else:
            color, alpha, edge_color, edge_alpha = 'lightgray', 0.1, 'gray', 0.2
        
        rect = plt.Rectangle((bin_lower, bin_lower), bin_upper - bin_lower, bin_upper - bin_lower,
                           facecolor=color, alpha=alpha, edgecolor=edge_color, 
                           linewidth=1, linestyle='--')
        ax1.add_patch(rect)
        
        ax1.axvline(bin_lower, color=edge_color, linestyle=':', alpha=edge_alpha, linewidth=1)
        ax1.axhline(bin_lower, color=edge_color, linestyle=':', alpha=edge_alpha, linewidth=1)
        
        if has_data:
            center_x = (bin_lower + bin_upper) / 2
            center_y = (bin_lower + bin_upper) / 2
            ax1.text(center_x, center_y, f'Bin {i+1}', ha='center', va='center', 
                    fontsize=7, alpha=0.6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Perfect calibration line for both plots
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.8, linewidth=2, zorder=4)
    ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.8, linewidth=2, zorder=4)
    
    # Raw calibration points
    if bin_counts_raw:
        distances_raw = [abs(conf - acc) for conf, acc in zip(bin_confidences_raw, bin_accuracies_raw)]
        colors_raw = ['red' if d > 0.1 else 'orange' if d > 0.05 else 'green' for d in distances_raw]
        
        ax1.scatter(bin_confidences_raw, bin_accuracies_raw, 
                   s=[count/max(bin_counts_raw)*300 + 100 for count in bin_counts_raw], 
                   alpha=0.8, c=colors_raw, edgecolors='black', linewidth=2)
        
        for conf, acc, count in zip(bin_confidences_raw, bin_accuracies_raw, bin_counts_raw):
            ax1.annotate(f'{count}', (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Calibrated plot
    bin_confidences_cal = []
    bin_accuracies_cal = []
    bin_counts_cal = []
    
    # Track which bins have data for calibrated model
    cal_bins_with_data = set()
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (calibrated_confidences > bin_lower) & (calibrated_confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_confidences_cal.append(calibrated_confidences[in_bin].mean())
            bin_accuracies_cal.append(calibrated_accuracies[in_bin].mean())
            bin_counts_cal.append(in_bin.sum())
            cal_bins_with_data.add((bin_lower, bin_upper))
    
    # Draw bucket regions for calibrated model (ax2)
    for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
        has_data = (bin_lower, bin_upper) in cal_bins_with_data
        
        if has_data:
            color, alpha, edge_color, edge_alpha = 'lightgreen', 0.15, 'green', 0.3
        else:
            color, alpha, edge_color, edge_alpha = 'lightgray', 0.1, 'gray', 0.2
        
        rect = plt.Rectangle((bin_lower, bin_lower), bin_upper - bin_lower, bin_upper - bin_lower,
                           facecolor=color, alpha=alpha, edgecolor=edge_color, 
                           linewidth=1, linestyle='--')
        ax2.add_patch(rect)
        
        ax2.axvline(bin_lower, color=edge_color, linestyle=':', alpha=edge_alpha, linewidth=1)
        ax2.axhline(bin_lower, color=edge_color, linestyle=':', alpha=edge_alpha, linewidth=1)
        
        if has_data:
            center_x = (bin_lower + bin_upper) / 2
            center_y = (bin_lower + bin_upper) / 2
            ax2.text(center_x, center_y, f'Bin {i+1}', ha='center', va='center', 
                    fontsize=7, alpha=0.6, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))
    
    # Calibrated points
    if bin_counts_cal:
        distances_cal = [abs(conf - acc) for conf, acc in zip(bin_confidences_cal, bin_accuracies_cal)]
        colors_cal = ['red' if d > 0.1 else 'orange' if d > 0.05 else 'green' for d in distances_cal]
        
        ax2.scatter(bin_confidences_cal, bin_accuracies_cal, 
                   s=[count/max(bin_counts_cal)*300 + 100 for count in bin_counts_cal], 
                   alpha=0.8, c=colors_cal, edgecolors='black', linewidth=2)
        
        for conf, acc, count in zip(bin_confidences_cal, bin_accuracies_cal, bin_counts_cal):
            ax2.annotate(f'{count}', (conf, acc), xytext=(5, 5), 
                        textcoords='offset points', fontsize=10, fontweight='bold')
    
    # Styling for both plots
    for ax, title_suffix, confidences, accuracies in [
        (ax1, "Before Calibration", raw_confidences, raw_accuracies),
        (ax2, f"After {calibrator_name}", calibrated_confidences, calibrated_accuracies)
    ]:
        ax.set_xlabel('Mean Predicted Confidence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Actual Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(f'{method_name} - {title_suffix}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add ECE info
        ece = expected_calibration_error(confidences, accuracies, n_bins)
        if ece < 0.05:
            interpretation = "Excellent"
            color = 'green'
        elif ece < 0.10:
            interpretation = "Good"
            color = 'orange'
        else:
            interpretation = "Poor"
            color = 'red'
        
        ax.text(0.05, 0.95, f'ECE: {ece:.3f}\n({interpretation})', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color, linewidth=2),
                fontsize=11, verticalalignment='top')
    
    # Overall title
    fig.suptitle(f'Calibration Improvement: {method_name} Method', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        # Only show if we're in an interactive environment
        try:
            plt.show()
        except:
            pass  # Ignore errors in headless environments
    
    return fig


def plot_confidence_histogram(confidences: np.ndarray, accuracies: np.ndarray,
                            n_bins: int = 20, title: str = "Confidence Distribution",
                            save_path: Optional[str] = None, show: bool = False) -> plt.Figure:
    """
    Plot histogram of confidence scores, separated by correctness.
    
    Shows the distribution of confidence for correct vs incorrect predictions.
    Well-calibrated models should show higher confidence for correct predictions.
    
    Args:
        confidences: Array of confidence scores [0, 1]
        accuracies: Array of binary correctness [0, 1]
        n_bins: Number of histogram bins
        title: Plot title
        save_path: Optional path to save the plot
        show: Whether to display the plot
        
    Returns:
        matplotlib Figure object
    """
    correct_confidences = confidences[accuracies == 1]
    incorrect_confidences = confidences[accuracies == 0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(correct_confidences, bins=n_bins, alpha=0.7, label='Correct Predictions', 
            color='green', density=True)
    ax.hist(incorrect_confidences, bins=n_bins, alpha=0.7, label='Incorrect Predictions', 
            color='red', density=True)
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"""Correct: μ={np.mean(correct_confidences):.3f}, σ={np.std(correct_confidences):.3f}
Incorrect: μ={np.mean(incorrect_confidences):.3f}, σ={np.std(incorrect_confidences):.3f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        # Only show if we're in an interactive environment
        try:
            plt.show()
        except:
            pass  # Ignore errors in headless environments
    
    return fig


def compare_calibration(confidences_before: np.ndarray, accuracies_before: np.ndarray,
                       confidences_after: np.ndarray, accuracies_after: np.ndarray,
                       method_name: str = "Calibration Method") -> Dict[str, Dict[str, float]]:
    """
    Compare calibration metrics before and after calibration.
    
    Args:
        confidences_before: Raw confidence scores [0, 1]
        accuracies_before: Binary correctness for raw scores [0, 1]
        confidences_after: Calibrated confidence scores [0, 1]
        accuracies_after: Binary correctness for calibrated scores [0, 1]
        method_name: Name of calibration method used
        
    Returns:
        Dictionary with before/after metrics and improvements
    """
    before_metrics = calibration_metrics(confidences_before, accuracies_before)
    after_metrics = calibration_metrics(confidences_after, accuracies_after)
    
    improvements = {}
    for metric in before_metrics:
        if metric in ['ECE', 'MCE', 'Brier Score', 'Reliability']:
            # Lower is better for these metrics
            improvement = before_metrics[metric] - after_metrics[metric]
            improvements[metric] = improvement
        else:
            # For other metrics, just show the change
            improvements[metric] = after_metrics[metric] - before_metrics[metric]
    
    return {
        'Before': before_metrics,
        'After': after_metrics,
        'Improvement': improvements,
        'Method': method_name
    }


def print_calibration_report(metrics: Dict[str, float], title: str = "Calibration Report"):
    """
    Print a formatted calibration metrics report.
    
    Args:
        metrics: Dictionary of calibration metrics
        title: Report title
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric:20s}: {value:.4f}")
        else:
            print(f"{metric:20s}: {value}")
    
    # Interpretation
    print("\nInterpretation:")
    if 'ECE' in metrics:
        ece = metrics['ECE']
        if ece < 0.05:
            print(f"  ECE ({ece:.3f}): Excellent calibration")
        elif ece < 0.10:
            print(f"  ECE ({ece:.3f}): Good calibration")
        elif ece < 0.20:
            print(f"  ECE ({ece:.3f}): Moderate calibration")
        else:
            print(f"  ECE ({ece:.3f}): Poor calibration")


if __name__ == "__main__":
    # Example usage with synthetic data
    np.random.seed(42)
    
    # Simulate poorly calibrated confidence scores (overconfident)
    n_samples = 1000
    raw_confidences = np.random.beta(3, 1, n_samples)  # Biased toward high confidence
    true_accuracies = (np.random.random(n_samples) < raw_confidences * 0.7).astype(int)  # Lower actual accuracy
    
    print("Example: Evaluating poorly calibrated confidence scores")
    
    # Calculate metrics
    metrics = calibration_metrics(raw_confidences, true_accuracies)
    print_calibration_report(metrics, "Raw Confidence Scores")
    
    # Plot reliability diagram
    plot_reliability_diagram(raw_confidences, true_accuracies, 
                           title="Example: Overconfident Model")
    
    # Plot confidence distribution
    plot_confidence_histogram(raw_confidences, true_accuracies,
                             title="Example: Confidence Distribution")
