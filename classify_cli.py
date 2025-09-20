#!/usr/bin/env python3
"""
Simple CLI classifier for coding agents to use when generating dataset examples.

Usage:
    python classify_cli.py "Your text here"
    
Returns:
    prediction: positive/negative/neutral
    confidence: 0.0-1.0
"""

import sys
import argparse
from logprobs_confidence import TransformerLogprobsClassifier


def main():
    parser = argparse.ArgumentParser(description='Classify text sentiment with confidence')
    parser.add_argument('text', help='Text to classify')
    parser.add_argument('--quiet', '-q', action='store_true', help='Only output prediction and confidence')
    
    args = parser.parse_args()
    
    if not args.text.strip():
        print("Error: Empty text provided")
        sys.exit(1)
    
    try:
        # Initialize classifier with confidence scoring
        if not args.quiet:
            print("Loading classifier...")
        
        classifier = TransformerLogprobsClassifier()
        
        # Classify the text with confidence
        result = classifier.get_real_logprobs_confidence(args.text)
        
        prediction = result.get('prediction', 'unknown')
        confidence = result.get('confidence', 0.0)
        
        if args.quiet:
            # Simple output for agents
            print(f"{prediction} {confidence}")
        else:
            # Human-readable output
            print(f"\nText: {args.text}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence}")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
