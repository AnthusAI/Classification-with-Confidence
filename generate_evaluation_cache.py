#!/usr/bin/env python3
"""
Generate Evaluation Cache

This script runs model evaluations ONCE and saves all results to cache files.
All visualization scripts will then use this cached data instead of re-running evaluations.

Usage:
    python generate_evaluation_cache.py

Output:
    - evaluation_cache/base_model_results.json
    - evaluation_cache/finetuned_model_results.json
    - evaluation_cache/calibrated_results.json
    - evaluation_cache/evaluation_metadata.json
"""

import json
import os
import random
import numpy as np
from datetime import datetime
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from logprobs_confidence import TransformerLogprobsClassifier
from calibration_metrics import expected_calibration_error, maximum_calibration_error

def load_test_data(n_samples: int = 1000) -> list:
    """Load and sample test data."""
    print(f"📂 Loading test set and sampling {n_samples} examples...")
    
    with open('fine_tuned_sentiment_model/test_set.json', 'r') as f:
        test_examples = json.load(f)
    
    # Sample randomly with fixed seed for reproducibility
    random.seed(42)
    sample_examples = random.sample(test_examples, min(n_samples, len(test_examples)))
    
    print(f"✅ Loaded {len(sample_examples)} examples")
    return sample_examples

def evaluate_model(classifier, examples: list, model_name: str) -> list:
    """Evaluate a model on examples and return detailed results."""
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
            'category': example.get('category', 'unknown'),
            'logprobs': result.get('logprobs', {}),
            'raw_response': result.get('raw_response', '')
        })
    
    accuracy = np.mean([r['correct'] for r in results])
    mean_confidence = np.mean([r['confidence'] for r in results])
    print(f"✅ {model_name}: {accuracy:.1%} accuracy, {mean_confidence:.1%} mean confidence")
    
    return results

def apply_calibration_methods(results: list) -> dict:
    """Apply calibration methods to model results."""
    print("🔧 Applying calibration methods...")
    
    confidences = np.array([r['confidence'] for r in results])
    accuracies = np.array([r['correct'] for r in results])
    
    # Platt Scaling
    platt_model = LogisticRegression()
    platt_model.fit(confidences.reshape(-1, 1), accuracies)
    platt_calibrated = platt_model.predict_proba(confidences.reshape(-1, 1))[:, 1]
    
    # Isotonic Regression
    isotonic_model = IsotonicRegression(out_of_bounds='clip')
    isotonic_calibrated = isotonic_model.fit_transform(confidences, accuracies)
    
    return {
        'platt_calibrated': platt_calibrated.tolist(),
        'isotonic_calibrated': isotonic_calibrated.tolist(),
        'platt_model_params': {
            'coef': platt_model.coef_.tolist(),
            'intercept': platt_model.intercept_.tolist()
        }
    }

def calculate_metrics(results: list, calibrated_data: dict = None) -> dict:
    """Calculate comprehensive metrics for results."""
    confidences = np.array([r['confidence'] for r in results])
    accuracies = np.array([r['correct'] for r in results])
    
    metrics = {
        'accuracy': float(np.mean(accuracies)),
        'mean_confidence': float(np.mean(confidences)),
        'ece': float(expected_calibration_error(confidences, accuracies, 20)),
        'mce': float(maximum_calibration_error(confidences, accuracies, 20)),
        'confidence_range': {
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        },
        'total_samples': len(results)
    }
    
    # Add calibrated metrics if available
    if calibrated_data:
        platt_cal = np.array(calibrated_data['platt_calibrated'])
        isotonic_cal = np.array(calibrated_data['isotonic_calibrated'])
        
        metrics['calibrated_metrics'] = {
            'platt_ece': float(expected_calibration_error(platt_cal, accuracies, 20)),
            'platt_mce': float(maximum_calibration_error(platt_cal, accuracies, 20)),
            'isotonic_ece': float(expected_calibration_error(isotonic_cal, accuracies, 20)),
            'isotonic_mce': float(maximum_calibration_error(isotonic_cal, accuracies, 20))
        }
    
    return metrics

def main():
    """Generate comprehensive evaluation cache."""
    print("🚀 Generating Evaluation Cache")
    print("=" * 50)
    print("This will run evaluations ONCE and cache all results for fast visualization generation.")
    print()
    
    # Create cache directory
    os.makedirs('evaluation_cache', exist_ok=True)
    
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
    
    # Apply calibration methods to both models
    print("🔧 Applying calibration methods to base model...")
    base_calibrated = apply_calibration_methods(base_results)
    
    print("🔧 Applying calibration methods to fine-tuned model...")
    ft_calibrated = apply_calibration_methods(ft_results)
    
    # Calculate comprehensive metrics
    print("📊 Calculating comprehensive metrics...")
    base_metrics = calculate_metrics(base_results, base_calibrated)
    ft_metrics = calculate_metrics(ft_results, ft_calibrated)
    
    # Save all results to cache
    print("💾 Saving results to cache...")
    
    # Base model results
    base_cache = {
        'results': base_results,
        'calibrated': base_calibrated,
        'metrics': base_metrics,
        'model_info': {
            'name': 'meta-llama/Llama-3.1-8B-Instruct',
            'type': 'base'
        }
    }
    
    with open('evaluation_cache/base_model_results.json', 'w') as f:
        json.dump(base_cache, f, indent=2)
    
    # Fine-tuned model results
    ft_cache = {
        'results': ft_results,
        'calibrated': ft_calibrated,
        'metrics': ft_metrics,
        'model_info': {
            'name': 'meta-llama/Llama-3.1-8B-Instruct',
            'type': 'fine_tuned',
            'path': './fine_tuned_sentiment_model'
        }
    }
    
    with open('evaluation_cache/finetuned_model_results.json', 'w') as f:
        json.dump(ft_cache, f, indent=2)
    
    # Metadata
    metadata = {
        'generated_at': datetime.now().isoformat(),
        'test_samples': len(test_examples),
        'random_seed': 42,
        'models_evaluated': ['base', 'fine_tuned'],
        'calibration_methods': ['platt_scaling', 'isotonic_regression'],
        'cache_files': [
            'base_model_results.json',
            'finetuned_model_results.json'
        ]
    }
    
    with open('evaluation_cache/evaluation_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\\n🎉 Evaluation cache generated successfully!")
    print("📁 Cache files created:")
    print("  • evaluation_cache/base_model_results.json")
    print("  • evaluation_cache/finetuned_model_results.json") 
    print("  • evaluation_cache/evaluation_metadata.json")
    print()
    print("📊 Summary:")
    print(f"  • Base Model: {base_metrics['accuracy']:.1%} accuracy, ECE: {base_metrics['ece']:.3f}")
    print(f"  • Fine-Tuned: {ft_metrics['accuracy']:.1%} accuracy, ECE: {ft_metrics['ece']:.3f}")
    print()
    print("✅ All visualization scripts can now use this cached data!")
    print("⚡ No more waiting for model evaluations during chart generation!")

if __name__ == "__main__":
    main()
