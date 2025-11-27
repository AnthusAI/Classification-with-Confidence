#!/usr/bin/env python3
"""
SageMaker Training Entry Point

This script runs inside the SageMaker training container and orchestrates the training process.
It reuses the existing SentimentFineTuner class to ensure identical behavior between local and
SageMaker training.

SageMaker Environment Variables:
- SM_MODEL_DIR: Where to save the trained model (/opt/ml/model)
- SM_CHANNEL_TRAINING: Where training data is mounted (/opt/ml/input/data/training)
- SM_OUTPUT_DATA_DIR: Where to write additional output data (/opt/ml/output/data)
- SM_NUM_GPUS: Number of GPUs available

Hyperparameters (passed via --sagemaker-program flag):
- model_name: HuggingFace model ID
- classification_mode: 'first-token' or 'last-token'
"""

import os
import sys
import argparse
from pathlib import Path

# Import the existing fine-tuning logic - NO CODE DUPLICATION!
from fine_tune_model import SentimentFineTuner


def parse_args():
    """
    Parse SageMaker hyperparameters passed via command line.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='SageMaker training entry point')

    # Model configuration
    parser.add_argument(
        '--model_name',
        type=str,
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='HuggingFace model ID to fine-tune'
    )

    # SageMaker directories (defaults from environment variables)
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'),
        help='Directory to save trained model'
    )

    parser.add_argument(
        '--train_dir',
        type=str,
        default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'),
        help='Directory containing training data'
    )

    parser.add_argument(
        '--output_data_dir',
        type=str,
        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'),
        help='Directory for additional output data'
    )

    # Training configuration
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=int(os.environ.get('SM_NUM_GPUS', '1')),
        help='Number of GPUs available'
    )

    return parser.parse_args()


def setup_dataset_symlink(train_dir: str) -> None:
    """
    Create a symlink from 'dataset/' to the SageMaker training data directory.

    SentimentFineTuner expects to find training data in a 'dataset/' directory
    in the current working directory. SageMaker mounts training data at
    SM_CHANNEL_TRAINING (/opt/ml/input/data/training).

    This function creates a symlink to bridge the gap.

    Args:
        train_dir: Path to SageMaker training data directory

    Raises:
        Exception: If symlink creation fails
    """
    dataset_link = Path('dataset')
    train_path = Path(train_dir)

    # Remove existing symlink if present
    if dataset_link.is_symlink():
        dataset_link.unlink()
        print(f"Removed existing symlink: dataset")

    # Remove existing directory if present (shouldn't happen in SageMaker)
    elif dataset_link.exists():
        print(f"Warning: 'dataset' exists as a real directory, not a symlink")
        return

    # Create symlink
    try:
        dataset_link.symlink_to(train_path)
        print(f"✓ Created symlink: dataset -> {train_path}")

        # Verify symlink works
        if dataset_link.exists() and dataset_link.is_dir():
            file_count = len(list(dataset_link.glob('*.txt')))
            print(f"✓ Found {file_count} data files in dataset/")
        else:
            raise Exception("Symlink created but dataset/ is not accessible")

    except Exception as e:
        raise Exception(f"Failed to create dataset symlink: {e}")


def main():
    """
    Main entry point for SageMaker training.

    This function:
    1. Parses SageMaker hyperparameters
    2. Sets up the dataset symlink
    3. Calls the existing SentimentFineTuner class
    4. Ensures output is saved to SM_MODEL_DIR
    """
    print("=" * 60)
    print("SageMaker Training Entry Point")
    print("=" * 60)

    # Parse arguments
    args = parse_args()

    print("\nConfiguration:")
    print(f"  Model: {args.model_name}")
    print(f"  Training data: {args.train_dir}")
    print(f"  Model output: {args.model_dir}")
    print(f"  GPUs: {args.num_gpus}")
    print()

    # Set up dataset symlink
    try:
        setup_dataset_symlink(args.train_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Training Mode: Sentiment Classification")
    print(f"{'='*60}\n")

    # Use existing SentimentFineTuner class - ZERO CODE DUPLICATION!
    # This ensures SageMaker training produces identical results to local training
    try:
        fine_tuner = SentimentFineTuner(
            model_name=args.model_name,
            output_dir=args.model_dir  # SageMaker will package everything in this directory
        )

        # Run training using existing logic
        print("Starting fine-tuning with existing SentimentFineTuner...")
        trainer = fine_tuner.fine_tune()

        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print("=" * 60)
        print(f"Model saved to: {args.model_dir}")

        # Verify required output files exist
        print("\nVerifying output files...")
        required_files = [
            'adapter_config.json',
            'adapter_model.safetensors',
            'test_set.json',
            'training_info.json'
        ]

        all_present = True
        for filename in required_files:
            filepath = Path(args.model_dir) / filename
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                print(f"  ✓ {filename} ({size_kb:.1f} KB)")
            else:
                print(f"  ✗ {filename} (MISSING)")
                all_present = False

        if not all_present:
            print("\nWARNING: Some expected files are missing!")
            sys.exit(1)

        print("\n✓ All required files present!")
        print("\nSageMaker will now package these files to S3.")

    except Exception as e:
        print(f"\n{'='*60}")
        print("Training Failed!")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
