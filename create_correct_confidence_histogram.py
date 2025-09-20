#!/usr/bin/env python3
"""
Create the CORRECT raw confidence distribution histogram using actual base model measurements.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import subprocess
import time

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_confidence_from_cli(text):
    """Get confidence score using our CLI tool."""
    try:
        result = subprocess.run(
            ['python', 'classify_cli.py', '--quiet', text],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            if len(parts) >= 2:
                return float(parts[1])
    except Exception as e:
        print(f"Error processing text: {e}")
    return None

def load_examples_from_files():
    """Load examples from all dataset files."""
    examples = []
    
    # Load from each category
    categories = [
        ('strong_positive', 'dataset/strong_positive.txt'),
        ('strong_negative', 'dataset/strong_negative.txt'),
        ('medium_positive', 'dataset/medium_positive.txt'),
        ('medium_negative', 'dataset/medium_negative.txt'),
        ('weak_positive', 'dataset/weak_positive.txt'),
        ('weak_negative', 'dataset/weak_negative.txt'),
    ]
    
    for category, filepath in categories:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                # Take a sample from each category for speed
                sample_size = min(50, len(lines))  # 50 examples per category = 300 total
                sampled_lines = lines[:sample_size]
                for line in sampled_lines:
                    examples.append((line, category))
                print(f"Loaded {len(sampled_lines)} examples from {category}")
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
    
    return examples

def create_correct_confidence_histogram():
    """Create the correct confidence histogram using actual measurements."""
    print("📊 Creating CORRECT confidence histogram with actual base model measurements...")
    
    # Load examples from dataset files
    examples = load_examples_from_files()
    print(f"Total examples to process: {len(examples)}")
    
    confidences = []
    categories = []
    
    print("🔍 Measuring confidence scores (this will take several minutes)...")
    
    for i, (text, category) in enumerate(examples):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(examples)} ({i/len(examples)*100:.1f}%)")
        
        confidence = get_confidence_from_cli(text)
        if confidence is not None:
            confidences.append(confidence)
            categories.append(category)
        
        # Small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    print(f"✅ Successfully measured {len(confidences)} confidence scores")
    
    # Create histogram
    plt.figure(figsize=(12, 8))
    
    # Create overall histogram
    plt.subplot(2, 1, 1)
    plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Examples')
    plt.title('CORRECTED: Raw Confidence Distribution\nActual Base Model Measurements from Dataset')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_conf = np.mean(confidences)
    min_conf = np.min(confidences)
    max_conf = np.max(confidences)
    
    plt.axvline(mean_conf, color='red', linestyle='--', 
                label=f'Mean: {mean_conf:.1%}')
    
    plt.text(0.02, 0.98, 
             f'ACTUAL Statistics:\n'
             f'Mean: {mean_conf:.1%}\n'
             f'Min: {min_conf:.1%}\n'
             f'Max: {max_conf:.1%}\n'
             f'Samples: {len(confidences)}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.legend()
    
    # Create breakdown by category
    plt.subplot(2, 1, 2)
    
    # Group by category
    category_confidences = {}
    for conf, cat in zip(confidences, categories):
        if cat not in category_confidences:
            category_confidences[cat] = []
        category_confidences[cat].append(conf)
    
    # Plot each category
    colors = ['red', 'darkred', 'orange', 'darkorange', 'lightblue', 'blue']
    for i, (cat, confs) in enumerate(category_confidences.items()):
        plt.hist(confs, bins=15, alpha=0.6, label=f'{cat} (n={len(confs)})', 
                color=colors[i % len(colors)])
    
    plt.xlabel('Confidence Score')
    plt.ylabel('Number of Examples')
    plt.title('Confidence Distribution by Category')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    ensure_dir('images/calibration')
    plt.savefig('images/calibration/raw_confidence_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ CORRECTED confidence histogram saved!")
    
    # Print summary statistics
    print("\n📊 ACTUAL CONFIDENCE STATISTICS:")
    print(f"Overall: Mean={mean_conf:.1%}, Min={min_conf:.1%}, Max={max_conf:.1%}")
    
    for cat, confs in category_confidences.items():
        cat_mean = np.mean(confs)
        cat_min = np.min(confs)
        cat_max = np.max(confs)
        print(f"{cat}: Mean={cat_mean:.1%}, Min={cat_min:.1%}, Max={cat_max:.1%} (n={len(confs)})")
    
    return confidences, categories

def main():
    """Create the corrected confidence histogram."""
    print("🚀 Creating CORRECTED confidence histogram with actual measurements...")
    
    confidences, categories = create_correct_confidence_histogram()
    
    print(f"\n✅ CORRECTED histogram created with {len(confidences)} actual measurements!")
    print("📊 This shows the REAL confidence distribution from our base model.")

if __name__ == "__main__":
    main()


