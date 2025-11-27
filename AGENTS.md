# AGENTS.md - Project Implementation Plan

**NOTE**: This document is intended for AI agents working on this project. The `README.md` file is the user-facing documentation that explains the final product to humans who want to understand and use the code.

## Project Overview

This project demonstrates how to extract confidence scores from Large Language Models (LLMs) when used as classifiers. The core insight is that LLMs show higher response consistency on familiar concepts and lower consistency on unfamiliar ones.

## Implementation Strategy

### Core Approach
- **Confidence Through Consistency**: Measure model uncertainty by running multiple predictions and calculating agreement
- **Fine-Tuning for Calibration**: Use LoRA fine-tuning to improve confidence calibration on domain-specific tasks
- **Evaluation Methodology**: Proper train/test splits with held-out data for unbiased evaluation

### Technical Approach

#### 1. Confidence Through Consistency
Using Hugging Face Transformers, we can measure confidence through multiple approaches:
- **Multiple Queries**: Ask the model the same question 5-10 times
- **Response Agreement**: Higher agreement = higher confidence
- **Consistency Score**: Percentage of responses that agree with the majority prediction

#### 2. Test Dataset Design
- **8 Categories**: Strong/medium/weak positive/negative, plus neutral positive/negative contexts
- **10,000 Examples**: ~1k examples per category for comprehensive evaluation
- **Varied Difficulty**: From clear sentiment to subtle/ambiguous cases

#### 3. Fine-Tuning for Better Calibration
LoRA fine-tuning improves both accuracy and confidence calibration:
- **Memory Efficient**: Only train ~2% of model parameters
- **Fast Training**: Complete in 10-20 minutes on consumer GPUs
- **Better Calibration**: Model learns task-specific uncertainty patterns
- **GPU Support**: Automatic detection of MPS/CUDA with CPU fallback

## Complete Workflow for AI Agents

### 🚀 **ONE-COMMAND PIPELINE** (Recommended)
```bash
# Complete end-to-end pipeline (fine-tuning + analysis + all charts)
python run_complete_pipeline.py

# Options:
python run_complete_pipeline.py --skip-finetuning  # Skip if model exists (recommended - weights included!)
python run_complete_pipeline.py --force-retrain    # Force retrain model
python run_complete_pipeline.py --skip-images      # Skip chart generation
```

**💡 Important**: This repository includes everything for flexible development:
- **No GPU**: Use evaluation cache for instant visualizations
- **Limited GPU**: Run fine-tuning locally (10-20 minutes) then use for inference
- **Full GPU**: Complete pipeline available for experimentation

### 📋 **STEP-BY-STEP WORKFLOW** (Manual Control)

#### Step 1: Fine-Tuning (Required for Model Usage)
```bash
# Install dependencies (one-time setup)
pip install peft bitsandbytes datasets accelerate

# Run fine-tuning (takes 10-20 minutes on GPU, uses 10k examples)
python fine_tune_model.py
```
**What this does:**
- Loads 10,000 examples from `dataset/` (8 categories)
- 80/20 split: 8,000 for training, 2,000 held out for evaluation
- Saves test set to `fine_tuned_sentiment_model/test_set.json`
- Creates LoRA adapter weights in `fine_tuned_sentiment_model/`

**⚠️ Note**: This step is required since fine-tuned model weights cannot be included in the repository (640MB exceeds GitHub's 100MB limit). The fine-tuning process takes 10-20 minutes on a modern GPU and needs to be run locally.

#### Step 2: Generate Evaluation Cache (Efficiency)
```bash
# Run model evaluations ONCE and cache results (takes ~30 minutes)
python generate_evaluation_cache.py
```
**What this does:**
- Loads both base and fine-tuned models
- Evaluates 1,000 samples from held-out test set
- Applies Platt scaling and Isotonic regression calibration
- Saves all results to `evaluation_cache/` for instant chart generation

#### Step 3: Generate All Charts
```bash
# Generate ALL visualization charts using cached data
python generate_all_charts.py
```
**What this does:**
- Loads cached evaluation data (no model loading!)
- Generates 12+ charts quickly
- Saves to `images/calibration/` and `images/fine_tuning/`

### 🔧 **INDIVIDUAL ANALYSIS SCRIPTS** (Advanced Usage)

#### Model Comparison
```bash
# Compare base vs fine-tuned performance (uses held-out test set)
python compare_base_vs_finetuned.py
```

#### Calibration Analysis
```bash
# Individual calibration method analysis
python calibration_demo.py
python business_reliability.py
python sample_size_thresholds.py
```

#### API Server (Optional)
```bash
# Start FastAPI server for efficient batch predictions
python classify_api.py &

# Test API with both models
./classify_fast.sh "This is amazing!" base
./classify_fast.sh "This is amazing!" finetuned
```

#### Educational Logprob Analysis Tool
```bash
# Show detailed token probability analysis (README format)
python logprob_demo_cli.py "I love this movie!"

# Compare base vs fine-tuned model
python logprob_demo_cli.py "Best worst thing ever" --model base
python logprob_demo_cli.py "Best worst thing ever" --model finetuned

# Show more tokens in the analysis
python logprob_demo_cli.py "This is okay" --top-k 10

# Simple output for scripting
python logprob_demo_cli.py "Some text" --quiet

# RAW PROMPT MODE - Pass text directly to model (not wrapped in classification)
python logprob_demo_cli.py "knock knock" --raw-prompt
python logprob_demo_cli.py "Once upon a time" --raw-prompt --model finetuned
```
**What this shows:**
- Raw log-probabilities (what the model actually computes)
- Converted probabilities and percentages
- Token ranking by probability
- Sentiment aggregation logic (classification mode)
- Next token predictions (raw prompt mode)
- Educational explanations of confidence levels

### Technical Implementation Details

#### Fine-Tuning Process (`fine_tune_model.py`)
1. **Model Loading**: Loads Llama 3.1-8B-Instruct with automatic device detection
2. **LoRA Setup**: Applies Low-Rank Adaptation to attention and MLP layers
3. **Dataset Preparation**: Formats 10,000 examples (8 categories) for instruction tuning
4. **Train/Test Split**: Single 80/20 split (8,000 train, 2,000 held-out test)
5. **Training**: 3 epochs with gradient accumulation and evaluation
6. **Model Saving**: Saves adapter weights to `fine_tuned_sentiment_model/`
7. **Test Set Preservation**: Saves held-out examples to `test_set.json` for evaluation

#### GPU Support
- **Apple Silicon**: Automatic MPS detection and usage
- **NVIDIA**: CUDA support with memory optimization
- **CPU Fallback**: Graceful degradation if GPU unavailable
- **Single Code Path**: No separate GPU/CPU versions needed

#### Model Comparison (`compare_base_vs_finetuned.py`)
1. **Loads both models**: Base Llama 3.1 and fine-tuned version
2. **Runs evaluation**: Tests on validation dataset
3. **Calculates metrics**: ECE, MCE, accuracy, confidence distributions
4. **Generates visualizations**: Saves charts to `images/fine_tuning/`
5. **Business analysis**: Threshold analysis for automated decisions

## Technical Details

### Hardware Flexibility for Agents

**🎨 Visualization/Analysis Only**:
- Dependencies: `matplotlib`, `numpy`, `scikit-learn`
- Hardware: Any machine (no GPU needed)
- Use cases: Generate charts, analyze cached results, modify visualizations

**🔍 Model Inference**:
- Dependencies: `torch`, `transformers` + analysis tools
- Hardware: 8GB+ GPU recommended (CPU fallback available)
- Use cases: Run predictions with included fine-tuned weights, evaluate on test data

**🚀 Full Development**:
- Dependencies: Full stack (`peft`, `bitsandbytes`, `datasets`, `accelerate`)
- Hardware: 16GB+ GPU for fine-tuning
- Use cases: Experiment with training parameters, modify fine-tuning approach

### What's Pre-Computed in Repository
- **Evaluation cache**: All model results pre-computed (~780KB)
- **Dataset**: Complete training and test datasets
- **Codebase**: All scripts for reproduction

**What's NOT included** (GitHub file size limits):
- **LoRA adapter**: Must be generated locally via fine-tuning (10-20 minutes)
- **Base Llama weights**: Downloads automatically via Hugging Face (subject to Meta's license)

**This enables agents to**: Work at visualization level immediately, or run complete pipeline locally

### Current File Structure (Updated)
```
# Dataset (10,000 examples total)
dataset/
├── strong_positive.txt      # 1,000 clear positive examples
├── strong_negative.txt      # 1,000 clear negative examples
├── medium_positive.txt      # 1,000 moderate positive examples
├── medium_negative.txt      # 1,000 moderate negative examples
├── weak_positive.txt        # 1,000 subtle positive examples
├── weak_negative.txt        # 1,000 subtle negative examples
├── neutral_positive.txt     # 1,000 neutral sports examples (positive context)
├── neutral_negative.txt     # 1,000 neutral workplace examples (negative context)
└── README.md               # Dataset documentation

# Fine-tuned model artifacts
fine_tuned_sentiment_model/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # Fine-tuned weights (updated format)
├── test_set.json           # 2,000 held-out examples for evaluation
├── chat_template.jinja     # Model chat template
└── README.md               # Model documentation

# Evaluation cache (for fast chart generation)
evaluation_cache/
├── base_model_results.json      # Base model evaluation results
├── finetuned_model_results.json # Fine-tuned model evaluation results
├── calibrated_results.json     # Platt/Isotonic calibration results
└── evaluation_metadata.json    # Evaluation run metadata

# Generated visualizations
images/
├── calibration/
│   ├── raw_confidence_histogram.png
│   ├── reliability_raw_model_output.png
│   ├── reliability_platt_scaling.png
│   ├── reliability_isotonic_regression.png
│   ├── business_reliability.png
│   └── business_reliability_progression.png
└── fine_tuning/
    ├── accuracy_comparison.png
    ├── calibration_error_comparison.png
    ├── confidence_distribution_changes.png
    ├── finetuning_raw_reliability_comparison.png
    ├── finetuning_platt_reliability_comparison.png
    └── finetuning_isotonic_reliability_comparison.png
```

### Confidence Calculation
```python
# Pseudocode
responses = []
for i in range(num_samples):
    response = model.classify(text)
    responses.append(response)

most_common = mode(responses)
confidence = count(most_common) / len(responses)
```


## 📁 **KEY SCRIPTS REFERENCE**

### Core Pipeline Scripts
- **`run_complete_pipeline.py`**: 🚀 **MAIN ORCHESTRATOR** - Runs entire pipeline
- **`fine_tune_model.py`**: 🎯 Fine-tuning with LoRA (Step 1 - Required)
- **`generate_evaluation_cache.py`**: 💾 Model evaluation caching (Step 2 - Efficiency)
- **`generate_all_charts.py`**: 📊 Chart generation (Step 3 - Visualization)

### Analysis Scripts
- **`compare_base_vs_finetuned.py`**: 🔄 Model performance comparison
- **`logprobs_confidence.py`**: 🧮 Core confidence calculation (DRY model loading)
- **`calibration_demo.py`**: 📈 Calibration method demonstrations
- **`business_reliability.py`**: 💼 Business decision analysis
- **`sample_size_thresholds.py`**: 📏 Sample size impact analysis

### Utility Scripts
- **`dataset_loader.py`**: 📂 Dataset loading and management
- **`calibration_metrics.py`**: 📊 ECE/MCE calculation functions
- **`classify_api.py`**: 🌐 FastAPI server (both base/fine-tuned models)
- **`classify_cli.py`**: 💻 Command-line classification interface
- **`logprob_demo_cli.py`**: 🎓 Educational tool showing detailed logprob analysis

### Legacy/Development Scripts (Can Ignore)
- **`main.py`**: Old main script (superseded by `run_complete_pipeline.py`)
- **`consistency_confidence.py`**: Alternative confidence method (not used in final)
- **`create_*.py`**: Old individual chart creation scripts (removed - use `generate_all_charts.py`)

## Agent Guidelines

### When Working on This Project

1. **Use the Pipeline**: Always start with `python run_complete_pipeline.py`
2. **Evaluation Cache**: Run `generate_evaluation_cache.py` once, then use `generate_all_charts.py` for iterations
3. **Held-Out Data**: All evaluations use the 2,000 held-out examples from `test_set.json`
4. **GPU Verification**: The code automatically detects and uses available GPUs (MPS/CUDA/CPU)
5. **Documentation**: Keep both technical (AGENTS.md) and user (README.md) docs updated
6. **DRY Principle**: All model loading goes through `TransformerLogprobsClassifier` class

### Code Style
- Clear, educational code (this is for learning, not production)
- Extensive comments explaining the confidence methodology
- Modular design for easy experimentation
- Progress indicators for long-running experiments

### 🔧 **CRITICAL TECHNICAL DETAILS**

#### Deterministic Generation (Temperature Fix)
- **Problem**: Llama 3.1 defaults to `do_sample=true`, `temperature=0.6`, `top_p=0.9`
- **Solution**: Set `do_sample=False`, `temperature=None`, `top_p=None` in `model.generate()`
- **Result**: Deterministic classification outputs, no warnings

#### Evaluation Methodology (Data Integrity)
- **Single Split**: One 80/20 split during fine-tuning (8k train, 2k test)
- **Held-Out Test**: All evaluations use the same 2,000 examples from `test_set.json`
- **No Data Leakage**: Test examples never seen during training
- **Proper Sampling**: Random sampling from held-out set, not full dataset

#### Model Loading Optimization (DRY Principle)
- **Unified Class**: `TransformerLogprobsClassifier` handles both base and fine-tuned models
- **Device Detection**: Automatic MPS/CUDA/CPU detection and placement
- **Memory Efficiency**: No `load_in_8bit` on MPS (incompatible)
- **Error Handling**: Robust fallbacks and detailed error messages

#### Visualization Pipeline (Performance)
- **Evaluation Cache**: Run expensive evaluations once, cache results
- **Fast Charts**: Generate 12+ charts in seconds using cached data
- **Consistent Scaling**: Shared Y-axes, circle sizes, color schemes
- **Square Aspects**: Reliability diagrams use `set_aspect('equal')` for 45° calibration line

#### Visual Styling and Consistency (Updated)
- **Seaborn Pastel Style**: All charts use `plt.style.use('seaborn-v0_8-pastel')` for consistent professional appearance
- **Dynamic Color System**: Colors extracted from style sheet via `plt.rcParams['axes.prop_cycle']`
- **Consistent Edge Colors**: All bars/dots use 30% darker version of face color for edges (no hard-coded black)
- **0-100% Scale**: All reliability charts show full confidence range, not just 50-100%
- **Below-50% Confidence**: Added explanation for why calibration can result in <50% confidence scores
- **Color Mapping**: Base model always uses `colors[0]`, fine-tuned model uses `colors[1]` from style sheet

#### Educational Tools (New)
- **Log-Probability CLI**: `logprob_demo_cli.py` shows detailed token probability analysis
- **Raw Prompt Mode**: `--raw-prompt` flag bypasses classification wrapper for direct model interaction
- **Model Comparison**: `--model base|finetuned` flag allows comparing both models
- **Educational Output**: Formatted tables showing log-probs, probabilities, and percentages
- **Business Context**: Explains confidence levels in terms of automation decisions

#### README Structure (Updated)
- **Business-Focused Introduction**: Explains LLMs as prediction machines using "Knock knock" example
- **Classification Section**: Shows high/low confidence examples with log-prob breakdowns
- **Algorithm Explanation**: 5-step process for computing "Total Probability" scores
- **Calibration Problem**: Explains why raw probabilities need adjustment, with below-50% confidence explanation
- **Business Decision Framework**: Comprehensive section on using confidence for automation decisions
- **Fine-Tuning Section**: Business-focused explanation of how fine-tuning increases automation rates
- **Sample Size Analysis**: Shows impact of evaluation dataset size on business decisions

### Validation Approach
1. **Pipeline Test**: Run `python run_complete_pipeline.py` end-to-end
2. **Cache Verification**: Confirm `evaluation_cache/` contains recent results
3. **Chart Generation**: Verify all 12+ charts generate without errors
4. **Model Comparison**: Validate fine-tuning improvements (ECE, accuracy)
5. **Calibration**: Verify Platt/Isotonic calibration works correctly

## 🚀 **AWS SAGEMAKER DEPLOYMENT** ✅

### Status: Successfully Deployed

This project can be deployed to production using **Amazon SageMaker Inference Components** with LoRA adapters for cost-efficient multi-adapter inference.

**Use Case**: Serve hundreds of customer-specific LoRA sentiment classifiers from a single endpoint using [SageMaker's multi-adapter inference capability](https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/).

### Why SageMaker Inference Components?

**Traditional Approach**:
- 1 endpoint per customer/model
- 1000 customers = 1000 endpoints = $$$$$
- Endpoint startup time: 5-10 minutes per customer
- Resource waste: Most endpoints idle most of the time

**Inference Components Approach**:
- 1 base model endpoint
- 1000+ LoRA adapters loaded dynamically
- Tiered caching: GPU (hot) → CPU (warm) → S3 (cold)
- Cost: ~$150/month vs ~$150,000/month (1000x savings)
- Startup: Base model once, adapters in <1 minute

### Deployment Guide

For complete deployment instructions, see **[docs/SAGEMAKER_DEPLOYMENT.md](docs/SAGEMAKER_DEPLOYMENT.md)**

**Quick Start:**
```bash
# 1. Fine-tune your model locally
python fine_tune_model.py

# 2. Package and upload LoRA adapter to S3
cd fine_tuned_sentiment_model
tar -czf ../sentiment_adapter.tar.gz adapter_model.safetensors adapter_config.json
aws s3 cp ../sentiment_adapter.tar.gz s3://your-bucket/adapters/

# 3. Deploy to SageMaker
export HF_TOKEN="your_huggingface_token"
python3 scripts/deploy_sagemaker.py
```

**Deployment time**: ~20-30 minutes
**Cost**: ~$1.25/hour (ml.g6e.xlarge with 48GB GPU)
**Capability**: Serve 10+ LoRA adapters from one endpoint

### Deployment Resources

**Scripts**:
- `scripts/deploy_sagemaker.py` - Complete deployment automation

**Documentation**:
- `docs/SAGEMAKER_DEPLOYMENT.md` - Complete deployment guide for multi-adapter inference
