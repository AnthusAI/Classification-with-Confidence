#!/usr/bin/env python3
"""
Regenerate All Visualizations with Updated Data

This script regenerates all visualizations using:
- 1,000 random samples from the held-out test set
- Proper base vs fine-tuned model comparisons
- Visible bucket divisions in calibration charts
- Updated business impact numbers
"""

import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import json
import random
import os
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from logprobs_confidence import TransformerLogprobsClassifier
from calibration_metrics import (
    expected_calibration_error, 
    maximum_calibration_error,
    plot_reliability_diagram
)

def load_test_data(n_samples: int = 1000) -> List[Dict]:
    """Load and sample test data."""
    print(f"📂 Loading test set and sampling {n_samples} examples...")
    
    with open('fine_tuned_sentiment_model/test_set.json', 'r') as f:
        test_examples = json.load(f)
    
    # Sample randomly with fixed seed for reproducibility
    random.seed(42)
    sample_examples = random.sample(test_examples, min(n_samples, len(test_examples)))
    
    print(f"✅ Loaded {len(sample_examples)} examples")
    return sample_examples

def evaluate_model(classifier, examples: List[Dict], model_name: str) -> List[Dict]:
    """Evaluate a model on examples."""
    print(f"📊 Evaluating {model_name} on {len(examples)} examples...")
    
    results = []
    for i, example in enumerate(examples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(examples)}")
        
        result = classifier.get_real_logprobs_confidence(example['text'])
        is_correct = (result['prediction'] == example['expected'])
        
        results.append({
            'text': example['text'],
            'expected': example['expected'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'correct': is_correct,
            'category': example.get('category', 'unknown')
        })
    
    accuracy = np.mean([r['correct'] for r in results])
    mean_confidence = np.mean([r['confidence'] for r in results])
    print(f"✅ {model_name}: {accuracy:.1%} accuracy, {mean_confidence:.1%} mean confidence")
    
    return results

def create_reliability_diagram_with_buckets(results: List[Dict], title: str, filename: str, n_bins: int = 20):
    """Create reliability diagram with visible bucket divisions."""
    confidences = np.array([r['confidence'] for r in results])
    accuracies = np.array([r['correct'] for r in results])
    
    # Calculate bin statistics manually
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if bin_lower == 0:  # Include 0 in first bin
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        
        if np.sum(in_bin) > 0:
            bin_accuracy = np.mean(accuracies[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            bin_count = np.sum(in_bin)
        else:
            bin_accuracy = 0
            bin_confidence = (bin_lower + bin_upper) / 2
            bin_count = 0
        
        bin_accuracies.append(bin_accuracy)
        bin_confidences.append(bin_confidence)
        bin_counts.append(bin_count)
    
    bin_accuracies = np.array(bin_accuracies)
    bin_confidences = np.array(bin_confidences)
    bin_counts = np.array(bin_counts)
    
    # Calculate ECE and MCE
    ece = expected_calibration_error(confidences, accuracies, n_bins)
    mce = maximum_calibration_error(confidences, accuracies, n_bins)
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Add visible bucket divisions as alternating stripes
    for i in range(n_bins):
        left = i / n_bins
        right = (i + 1) / n_bins
        if i % 2 == 0:  # Alternate stripe pattern
            ax.axvspan(left, right, alpha=0.1, color='lightblue', zorder=0)
    
    # Add bucket boundary lines
    for i in range(1, n_bins):
        ax.axvline(i / n_bins, color='lightgray', linestyle=':', alpha=0.7, zorder=1)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
    
    # Plot reliability curve
    valid_bins = bin_counts > 0
    if np.any(valid_bins):
        ax.plot(bin_confidences[valid_bins], bin_accuracies[valid_bins], 
                'o-', linewidth=3, markersize=8, color='#2E86AB', label='Model Calibration')
        
        # Add sample count annotations
        for conf, acc, count in zip(bin_confidences[valid_bins], 
                                   bin_accuracies[valid_bins], 
                                   bin_counts[valid_bins]):
            ax.annotate(f'{int(count)}', (conf, acc), xytext=(5, 5), 
                       textcoords='offset points', fontsize=9, alpha=0.8)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Mean Predicted Confidence', fontsize=14)
    ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=14)
    ax.set_title(f'{title}\\nECE: {ece:.3f}, MCE: {mce:.3f}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add bucket information
    ax.text(0.02, 0.98, f'{n_bins} buckets (5% each)\\nTotal samples: {len(results)}', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {filename}")

def create_confidence_histogram(results: List[Dict], title: str, filename: str):
    """Create confidence distribution histogram."""
    confidences = [r['confidence'] for r in results]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(confidences, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Color bars by confidence level
    for i, (patch, bin_left, bin_right) in enumerate(zip(patches, bins[:-1], bins[1:])):
        confidence_level = (bin_left + bin_right) / 2
        if confidence_level < 0.5:
            patch.set_facecolor('lightcoral')
        elif confidence_level < 0.8:
            patch.set_facecolor('gold')
        else:
            patch.set_facecolor('lightgreen')
    
    ax.set_xlabel('Confidence Score', fontsize=14)
    ax.set_ylabel('Number of Predictions', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conf = np.mean(confidences)
    ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_conf:.3f}')
    ax.legend()
    
    plt.tight_layout()
    os.makedirs('images/calibration', exist_ok=True)
    plt.savefig(f'images/calibration/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {filename}")

def create_side_by_side_comparison(base_results: List[Dict], ft_results: List[Dict], 
                                 title: str, filename: str):
    """Create side-by-side calibration comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Base model subplot
    base_confidences = np.array([r['confidence'] for r in base_results])
    base_accuracies = np.array([r['correct'] for r in base_results])
    base_ece = expected_calibration_error(base_confidences, base_accuracies, 20)
    base_mce = maximum_calibration_error(base_confidences, base_accuracies, 20)
    
    # Fine-tuned model subplot  
    ft_confidences = np.array([r['confidence'] for r in ft_results])
    ft_accuracies = np.array([r['correct'] for r in ft_results])
    ft_ece = expected_calibration_error(ft_confidences, ft_accuracies, 20)
    ft_mce = maximum_calibration_error(ft_confidences, ft_accuracies, 20)
    
    # Plot both with bucket stripes
    for ax, confidences, accuracies, model_name, ece, mce in [
        (ax1, base_confidences, base_accuracies, "Base Model", base_ece, base_mce),
        (ax2, ft_confidences, ft_accuracies, "Fine-Tuned Model", ft_ece, ft_mce)
    ]:
        # Add bucket stripes
        for i in range(20):
            left = i / 20
            right = (i + 1) / 20
            if i % 2 == 0:
                ax.axvspan(left, right, alpha=0.1, color='lightblue', zorder=0)
        
        # Bucket lines
        for i in range(1, 20):
            ax.axvline(i / 20, color='lightgray', linestyle=':', alpha=0.7, zorder=1)
        
        # Perfect calibration
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
        
        # Calculate and plot reliability curve
        bin_boundaries = np.linspace(0, 1, 21)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            if bin_lower == 0:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_confidences.append(np.mean(confidences[in_bin]))
                bin_accuracies.append(np.mean(accuracies[in_bin]))
                bin_counts.append(np.sum(in_bin))
        
        if bin_confidences:
            ax.plot(bin_confidences, bin_accuracies, 'o-', linewidth=3, markersize=8, 
                   color='#2E86AB', label='Model Calibration')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Mean Predicted Confidence', fontsize=14)
        ax.set_ylabel('Fraction of Positives (Accuracy)', fontsize=14)
        ax.set_title(f'{model_name}\\nECE: {ece:.3f}, MCE: {mce:.3f}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=18, fontweight='bold')
    plt.tight_layout()
    os.makedirs('images/fine_tuning', exist_ok=True)
    plt.savefig(f'images/fine_tuning/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved {filename}")

def main():
    """Regenerate all visualizations."""
    print("🎨 Regenerating All Visualizations with Updated Data")
    print("=" * 60)
    
    # Load test data
    test_examples = load_test_data(1000)
    
    # Load models
    print("⏳ Loading base model...")
    base_classifier = TransformerLogprobsClassifier('meta-llama/Llama-3.1-8B-Instruct')
    
    print("⏳ Loading fine-tuned model...")
    ft_classifier = TransformerLogprobsClassifier(
        model_name='meta-llama/Llama-3.1-8B-Instruct',
        fine_tuned_path='./fine_tuned_sentiment_model'
    )
    
    # Evaluate both models
    base_results = evaluate_model(base_classifier, test_examples, "Base Model")
    ft_results = evaluate_model(ft_classifier, test_examples, "Fine-Tuned Model")
    
    print("\\n🎨 Creating visualizations...")
    
    # 1. Individual calibration charts with visible buckets
    create_reliability_diagram_with_buckets(
        base_results, 
        "Base Model Reliability (Raw Output)", 
        "reliability_raw_model_output.png"
    )
    
    create_reliability_diagram_with_buckets(
        ft_results, 
        "Fine-Tuned Model Reliability", 
        "reliability_finetuned_model_output.png"
    )
    
    # 2. Confidence histograms
    create_confidence_histogram(
        base_results,
        "Base Model Confidence Distribution",
        "raw_confidence_histogram.png"
    )
    
    create_confidence_histogram(
        ft_results,
        "Fine-Tuned Model Confidence Distribution", 
        "finetuned_confidence_histogram.png"
    )
    
    # 3. Side-by-side comparisons
    create_side_by_side_comparison(
        base_results, ft_results,
        "Calibration Comparison: Base vs Fine-Tuned",
        "calibration_comparison_side_by_side.png"
    )
    
    print("\\n🎉 All visualizations regenerated successfully!")
    print("📁 Check images/calibration/ and images/fine_tuning/ directories")

if __name__ == "__main__":
    main()
