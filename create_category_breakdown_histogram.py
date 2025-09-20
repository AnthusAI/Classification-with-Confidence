#!/usr/bin/env python3
"""
Create confidence distribution by category histogram.
This shows the breakdown by dataset category (strong/medium/weak/neutral).
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
import random
from pathlib import Path

def ensure_dir(directory):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def get_confidence_from_api(text):
    """Get confidence score from the running API"""
    try:
        response = requests.get(
            'http://localhost:8000/classify',
            params={'text': text},
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data['confidence']
        else:
            print(f"API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error calling API: {e}")
        return None

def load_examples_from_file(filepath):
    """Load examples from a text file, filtering out comments and empty lines"""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    examples = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#'):
            examples.append(line)
    
    return examples

def main():
    print("🚀 Creating category breakdown confidence histogram...")
    
    # Categories with their file paths, total counts, and sample sizes
    categories = [
        ('Strong Positive', 'dataset/strong_positive.txt', 500, 25),
        ('Strong Negative', 'dataset/strong_negative.txt', 500, 25),
        ('Medium Positive', 'dataset/medium_positive.txt', 1000, 35),
        ('Medium Negative', 'dataset/medium_negative.txt', 1000, 35),
        ('Weak Positive', 'dataset/weak_positive.txt', 2000, 70),
        ('Weak Negative', 'dataset/weak_negative.txt', 2000, 70),
        ('Neutral Positive', 'dataset/neutral_positive.txt', 10, 10),
        ('Neutral Negative', 'dataset/neutral_negative.txt', 10, 10),
    ]
    
    category_data = {}
    
    for name, filepath, total_count, sample_size in categories:
        print(f"📊 Sampling {sample_size} examples from {name}...")
        
        examples = load_examples_from_file(filepath)
        if not examples:
            print(f"No examples found in {filepath}")
            continue
        
        # Sample examples
        if len(examples) > sample_size:
            sampled_examples = random.sample(examples, sample_size)
        else:
            sampled_examples = examples
        
        # Get confidence scores
        confidences = []
        for i, example in enumerate(sampled_examples):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(sampled_examples)}")
            
            confidence = get_confidence_from_api(example)
            if confidence is not None:
                confidences.append(confidence)
        
        print(f"   ✅ Got {len(confidences)} confidence scores from {name}")
        
        category_data[name] = {
            'confidences': confidences,
            'total_count': total_count,
            'sample_size': sample_size
        }
    
    # Create category breakdown histogram
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = ['red', 'darkred', 'orange', 'darkorange', 'lightblue', 'blue', 'purple', 'magenta']
    for i, (name, data) in enumerate(category_data.items()):
        if data['confidences']:
            ax.hist(data['confidences'], bins=15, alpha=0.6, 
                    label=f'{name} (n={len(data["confidences"])})', 
                    color=colors[i % len(colors)])
    
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Number of Examples')
    ax.set_title('Confidence Distribution by Dataset Category')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    ensure_dir('images/calibration')
    plt.savefig('images/calibration/confidence_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Category breakdown histogram saved to images/calibration/confidence_by_category.png")
    
    # Print detailed statistics
    print("\n📊 DETAILED STATISTICS BY CATEGORY:")
    for name, data in category_data.items():
        if data['confidences']:
            confidences = data['confidences']
            mean_conf = np.mean(confidences)
            min_conf = np.min(confidences)
            max_conf = np.max(confidences)
            low_count = sum(1 for c in confidences if c < 0.7)
            high_count = sum(1 for c in confidences if c >= 0.9)
            total = len(confidences)
            
            print(f"{name}: Mean={mean_conf:.1%}, Range={min_conf:.1%}-{max_conf:.1%}, "
                  f"Low(<70%)={low_count}, High(≥90%)={high_count}, Total={total}")

if __name__ == "__main__":
    main()


