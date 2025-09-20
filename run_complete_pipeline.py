#!/usr/bin/env python3
"""
Complete Pipeline Orchestration Script

This script runs the entire confidence calibration pipeline:
1. Fine-tune the model with the expanded dataset
2. Compare base vs fine-tuned model performance
3. Generate all calibration analyses and visualizations
4. Create all missing images for the README

Usage:
    python run_complete_pipeline.py [--skip-finetuning] [--skip-images]
    
Options:
    --skip-finetuning: Skip fine-tuning if model already exists
    --skip-images: Skip image generation
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

def print_header(title, width=80):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(f" {title.center(width-2)} ")
    print("=" * width)

def print_section(title, width=80):
    """Print a formatted section header"""
    print("\n" + "-" * width)
    print(f" {title}")
    print("-" * width)

def run_command(command, description, check_success=True):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"📝 Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check_success, 
                              capture_output=True, text=True)
        
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("⚠️ Errors:")
            print(result.stderr)
            
        if check_success and result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        elif not check_success:
            print(f"📝 {description} completed (return code: {result.returncode})")
            
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error output:", e.stderr)
        return False
    except Exception as e:
        print(f"❌ Unexpected error in {description}: {e}")
        return False

def check_api_server():
    """Check if API server is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_api_server():
    """Start the API server in background"""
    print("🚀 Starting API server...")
    
    # Kill any existing server
    run_command("pkill -f 'classify_api.py'", "Stopping any existing API server", check_success=False)
    time.sleep(2)
    
    # Start new server in background
    subprocess.Popen([sys.executable, "classify_api.py"], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL)
    
    # Wait for server to start
    print("⏳ Waiting for API server to start...")
    for i in range(30):  # Wait up to 30 seconds
        if check_api_server():
            print("✅ API server is running!")
            return True
        time.sleep(1)
        if i % 5 == 0:
            print(f"   Still waiting... ({i+1}/30 seconds)")
    
    print("❌ API server failed to start!")
    return False

def verify_dataset():
    """Verify dataset is loaded correctly"""
    print_section("Verifying Dataset")
    
    success = run_command(
        'python -c "from dataset_loader import DatasetLoader; loader = DatasetLoader(); examples = loader.load_all(); print(f\'Total examples: {len(examples)}\'); categories = {}; [categories.update({example.get(\'category\', \'unknown\'): categories.get(example.get(\'category\', \'unknown\'), 0) + 1}) for example in examples]; [print(f\'{cat}: {count}\') for cat, count in sorted(categories.items())]"',
        "Verifying dataset loading"
    )
    
    return success

def run_finetuning(force_retrain=False):
    """Run the fine-tuning process"""
    print_section("Fine-Tuning Model")
    
    # Check if fine-tuned model already exists
    if os.path.exists("fine_tuned_sentiment_model") and not force_retrain:
        print("📝 Fine-tuned model already exists!")
        print("⏭️ Skipping fine-tuning (use --force-retrain to override)")
        return True
    
    if os.path.exists("fine_tuned_sentiment_model"):
        print("🔄 Retraining model with expanded dataset...")
    
    success = run_command(
        "python fine_tune_model.py",
        "Fine-tuning Llama 3.1 with 10,000 examples"
    )
    
    return success

def run_model_comparison():
    """Run base vs fine-tuned model comparison"""
    print_section("Comparing Base vs Fine-Tuned Models")
    
    print("🎯 Using 1000 examples for comprehensive business impact analysis")
    success = run_command(
        "python compare_base_vs_finetuned.py",
        "Comparing model performance and generating charts"
    )
    
    return success

def run_calibration_analysis():
    """Run calibration analysis"""
    print_section("Calibration Analysis")
    
    success = run_command(
        "python calibration_demo.py",
        "Running calibration analysis and generating reliability diagrams"
    )
    
    return success

def generate_confidence_histogram():
    """Generate the confidence histogram"""
    print_section("Generating Confidence Histogram")
    
    success = run_command(
        "python create_realistic_confidence_histogram.py",
        "Creating confidence distribution histogram"
    )
    
    return success

def generate_missing_images():
    """Generate any missing visualizations"""
    print_section("Generating Missing Visualizations")
    
    # List of image generation scripts
    image_scripts = [
        ("create_missing_visualizations.py", "Creating missing README visualizations"),
        ("create_simple_missing_images.py", "Creating additional missing images"),
    ]
    
    success = True
    for script, description in image_scripts:
        if os.path.exists(script):
            script_success = run_command(f"python {script}", description)
            success = success and script_success
        else:
            print(f"⏭️ Skipping {script} (not found)")
    
    return success

def main():
    """Main pipeline orchestration"""
    parser = argparse.ArgumentParser(description="Run complete confidence calibration pipeline")
    parser.add_argument("--skip-finetuning", action="store_true", 
                       help="Skip fine-tuning if model already exists")
    parser.add_argument("--force-retrain", action="store_true",
                       help="Force retraining even if model exists")
    parser.add_argument("--skip-images", action="store_true",
                       help="Skip image generation")
    
    args = parser.parse_args()
    
    print_header("CONFIDENCE CALIBRATION PIPELINE")
    print("🎯 Running complete pipeline with 10,000 examples")
    print("📊 This will take 30-60 minutes depending on your hardware")
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Step 1: Verify dataset
    if not verify_dataset():
        print("❌ Dataset verification failed!")
        return 1
    
    # Step 2: Start API server
    if not check_api_server():
        if not start_api_server():
            print("❌ Failed to start API server!")
            return 1
    else:
        print("✅ API server is already running!")
    
    # Step 3: Fine-tuning (optional)
    if not args.skip_finetuning:
        if not run_finetuning(force_retrain=args.force_retrain):
            print("❌ Fine-tuning failed!")
            return 1
    else:
        print("⏭️ Skipping fine-tuning as requested")
    
    # Step 4: Model comparison
    if not run_model_comparison():
        print("❌ Model comparison failed!")
        return 1
    
    # Step 5: Calibration analysis
    if not run_calibration_analysis():
        print("❌ Calibration analysis failed!")
        return 1
    
    # Step 6: Generate visualizations
    if not args.skip_images:
        if not generate_confidence_histogram():
            print("❌ Confidence histogram generation failed!")
            return 1
        
        if not generate_missing_images():
            print("❌ Some image generation failed!")
            return 1
    else:
        print("⏭️ Skipping image generation as requested")
    
    # Final summary
    print_header("PIPELINE COMPLETE!")
    print("✅ All steps completed successfully!")
    print("\n📊 Generated outputs:")
    print("   - Fine-tuned model: fine_tuned_sentiment_model/")
    print("   - Comparison charts: images/fine_tuning/")
    print("   - Calibration charts: images/calibration/")
    print("   - README visualizations: images/")
    print("\n🎉 Your confidence calibration analysis is ready!")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
