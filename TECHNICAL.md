# Get Started: Run the Experiments Yourself!

Ready to dive in? This guide will get you up and running with confidence scoring in just a few commands. All examples and results in the main article are based on actual experiments conducted using Meta's Llama 3.1-8B-Instruct model—and now you can run them yourself!

## Quick Start

### ⚡ **INSTANT VISUALIZATION** (No Models Needed!)
```bash
# Just want to see all the charts? Clone and run:
git clone <your-repo>
cd "Classification with Confidence"
pip install matplotlib numpy scikit-learn
python generate_all_charts.py
```
This generates all 12+ visualization charts in seconds using pre-computed cached data. No GPU, no model downloads, no waiting!

### 🚀 **ONE-COMMAND PIPELINE** (Recommended)
```bash
# Complete end-to-end pipeline (fine-tuning + analysis + all charts)
python run_complete_pipeline.py

# Options:
python run_complete_pipeline.py --skip-finetuning  # Skip if model exists (recommended - weights included!)
python run_complete_pipeline.py --force-retrain    # Force retrain model
python run_complete_pipeline.py --skip-images      # Skip chart generation
```

**💡 Note**: Due to GitHub's 100MB file size limits, the fine-tuned model weights are not included in the repository. You'll need to run the fine-tuning step first, but don't worry - it only takes 10-20 minutes on a modern GPU!

**🚀 Super Fast Start**: Want to see all the visualizations instantly without any models? The evaluation cache is included, so you can run `python generate_all_charts.py` immediately after cloning - no GPU, no model downloads, no waiting!

### 📋 **STEP-BY-STEP WORKFLOW** (Manual Control)

#### Step 1: Fine-Tuning (Required for Model Usage)
```bash
# Install dependencies (one-time setup)
pip install peft bitsandbytes datasets accelerate

# Run fine-tuning (takes 10-20 minutes on GPU, uses 10k examples)
python fine_tune_model.py
```

#### Step 2: Generate Evaluation Cache (Efficiency)
```bash
# Run model evaluations ONCE and cache results (takes ~30 minutes)
python generate_evaluation_cache.py
```

#### Step 3: Generate All Charts
```bash
# Generate ALL visualization charts using cached data
python generate_all_charts.py
```

## Model Details

- **Base Model**: Meta Llama 3.1-8B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Dataset**: 10,000 sentiment examples across 8 categories
- **Hardware**: Supports Apple Silicon (MPS), NVIDIA (CUDA), and CPU
- **Memory Requirements**: 16GB+ GPU memory recommended

## Key Scripts

### Core Pipeline Scripts
- **`run_complete_pipeline.py`**: Main orchestrator - runs entire pipeline
- **`fine_tune_model.py`**: Fine-tuning with LoRA (Step 1 - Required)
- **`generate_evaluation_cache.py`**: Model evaluation caching (Step 2)
- **`generate_all_charts.py`**: Chart generation (Step 3)

### Analysis Scripts
- **`compare_base_vs_finetuned.py`**: Model performance comparison
- **`logprobs_confidence.py`**: Core confidence calculation
- **`calibration_demo.py`**: Calibration method demonstrations
- **`business_reliability.py`**: Business decision analysis

### Educational Tools
- **`logprob_demo_cli.py`**: Interactive tool showing detailed log-probability analysis
```bash
# Show detailed token probability analysis
python logprob_demo_cli.py "I love this movie!"

# Compare Llama 3.1-8B vs fine-tuned model
python logprob_demo_cli.py "Best worst thing ever" --model base
python logprob_demo_cli.py "Best worst thing ever" --model finetuned

# RAW PROMPT MODE - Pass text directly to model
python logprob_demo_cli.py "knock knock" --raw-prompt
```

## Dataset Structure

```
dataset/
├── strong_positive.txt      # 500 clear positive examples
├── strong_negative.txt      # 500 clear negative examples
├── medium_positive.txt      # 1,000 moderate positive examples
├── medium_negative.txt      # 1,000 moderate negative examples
├── weak_positive.txt        # 2,500 subtle positive examples
├── weak_negative.txt        # 2,500 subtle negative examples
├── neutral_positive.txt     # 1,000 neutral sports examples
└── neutral_negative.txt     # 1,000 neutral workplace examples
```

## Technical Requirements & Hardware Flexibility

### Choose Your Level of Participation

**🎨 Visualization Only** (No GPU needed):
```bash
pip install matplotlib numpy scikit-learn
```

**🔍 Model Inference** (8GB+ GPU recommended):
```bash
pip install torch transformers
pip install matplotlib numpy scikit-learn
```

**🚀 Full Pipeline** (16GB+ GPU for fine-tuning):
```bash
pip install torch transformers peft bitsandbytes datasets accelerate
pip install scikit-learn matplotlib numpy pandas
```

### Hardware Support
- **No GPU**: Visualizations and chart generation work perfectly
- **Limited GPU** (8GB+): Run pre-trained models, use included fine-tuned weights
- **Full GPU** (16GB+): Complete fine-tuning pipeline available
- **Apple Silicon**: Automatic MPS detection across all levels
- **NVIDIA**: CUDA support with memory optimization
- **CPU Fallback**: Graceful degradation for all components

### What's Included in This Repository
- **Evaluation cache**: Pre-computed results (~780KB) for instant visualization
- **Dataset**: 10,000 training examples + 2,000 held-out test examples
- **Complete codebase**: All scripts for fine-tuning, evaluation, and visualization

**What's NOT included** (due to GitHub's 100MB file size limits):
- **LoRA adapter weights**: You'll need to run fine-tuning locally (10-20 minutes)
- **Base Llama model**: Downloads automatically via Hugging Face (subject to Meta's license)

**This means you can**:
- Generate all visualizations instantly (evaluation cache included)
- Run complete fine-tuning pipeline locally (all code included)
- Reproduce all results from the article yourself

## Reproducing Specific Examples

### "Knock Knock" Example
```bash
python logprob_demo_cli.py "Knock knock." --raw-prompt --model base
```

### Sentiment Classification Examples
```bash
# High confidence example
python logprob_demo_cli.py "I love this movie!" --model base

# Low confidence example  
python logprob_demo_cli.py "Best worst thing ever" --model base

# Medium confidence example
python logprob_demo_cli.py "Not bad" --model base
```

## Evaluation Methodology

- **Single Split**: One 80/20 split during fine-tuning (8k train, 2k test)
- **Held-Out Test**: All evaluations use the same 2,000 examples from `test_set.json`
- **No Data Leakage**: Test examples never seen during training
- **Calibration Methods**: Platt scaling and Isotonic regression

## File Structure

```
# Fine-tuned model artifacts
fine_tuned_sentiment_model/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # Fine-tuned weights
├── test_set.json           # 2,000 held-out examples
└── README.md               # Model documentation

# Evaluation cache (for fast chart generation)
evaluation_cache/
├── base_model_results.json      # Llama 3.1-8B evaluation results
├── finetuned_model_results.json # Fine-tuned model results
└── calibrated_results.json     # Calibration results

# Generated visualizations
images/
├── calibration/            # Calibration analysis charts
└── fine_tuning/           # Fine-tuning comparison charts
```

## Performance Notes

- **Training Time**: 10-20 minutes on modern GPUs
- **Evaluation Time**: ~30 minutes for full evaluation cache generation
- **Chart Generation**: Seconds using cached results
- **Memory Usage**: Optimized for consumer hardware

**Ready to get started?** Jump in with the one-command pipeline above, or dive deeper into the individual scripts. Have questions? Feel free to explore the codebase or check out the AGENTS.md file for detailed implementation notes.

**Happy experimenting!** 🚀
