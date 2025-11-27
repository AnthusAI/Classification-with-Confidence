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

This project supports **two distinct deployment paths**:

- **Path 1: Local Development** - Train and test locally for research/development
- **Path 2: SageMaker Production** - Train on AWS, deploy to endpoint for production

Choose the path based on requirements:
- Use **Path 1** for experimentation, prototyping, or when you have local GPU
- Use **Path 2** for production deployments, scaling, or when you need AWS infrastructure

---

## PATH 1: LOCAL DEVELOPMENT WORKFLOW

### 📋 **STEP-BY-STEP** (Manual Control)

#### Step 1: Fine-Tuning

```bash
# Set HuggingFace token (required for Llama model access)
export HF_TOKEN="your_huggingface_token"

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
- Requires: 16GB+ GPU VRAM
- Duration: 10-20 minutes on GPU

**⚠️ Note**: The fine-tuning process generates LoRA adapter weights (~670MB). Training takes 10-20 minutes on a modern GPU.

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

---

## PATH 2: SAGEMAKER PRODUCTION WORKFLOW

### 📋 **COMPLETE SAGEMAKER PIPELINE**

#### Step 1: Train on SageMaker

```bash
# Set HuggingFace token (required for Llama model access)
export HF_TOKEN="your_huggingface_token"

# Train on SageMaker (no local GPU needed!)
python fine_tune_model.py --sagemaker --instance-type ml.g6e.2xlarge
```

**What this does:**
- Packages code and dataset, uploads to S3
- Submits SageMaker training job
- Monitors training progress with live logs
- Downloads trained model artifacts automatically
- **Cost**: $0.87 (~21 minutes on ml.g6e.2xlarge)
- **No local GPU required**: Training runs on AWS
- **Output**: Model saved to `fine_tuned_sentiment_model/`

#### Step 2: Deploy to SageMaker Endpoint

```bash
# Deploy endpoint with base model + fine-tuned adapter
python scripts/deploy_sagemaker.py
```

**What this does:**
- Creates SageMaker endpoint (~6 minutes)
- Deploys base Llama 3.1-8B model (~18 minutes)
- Attaches fine-tuned LoRA adapter (~1 minute)
- **Cost**: $1.25/hour while running (ml.g6e.xlarge)
- **Total deployment time**: ~25 minutes

#### Step 3: Test SageMaker Endpoint

```bash
# Test deployed endpoint
python scripts/test_endpoint.py <endpoint-name>

# Test with custom text
python scripts/test_endpoint.py <endpoint-name> --text "Your custom text"
```

**What this does:**
- Invokes SageMaker endpoint via boto3
- Tests with multiple sentiment examples
- Returns generated text predictions
- Verifies endpoint is working correctly

#### Step 4: Clean Up (Stop Costs)

```bash
# Delete endpoint to stop hourly costs
aws sagemaker delete-endpoint --endpoint-name <endpoint-name> --region us-east-1

# Endpoint stops charging immediately after deletion
```

**Important**: Endpoints cost $1.25/hour while running. Always delete when done testing!

### 📊 **SAGEMAKER COST BREAKDOWN**

| Operation | Instance | Duration | Cost |
|-----------|----------|----------|------|
| Training | ml.g6e.2xlarge | ~21 min | $0.87 |
| Endpoint (running) | ml.g6e.xlarge | Per hour | $1.25/hr |
| **Total for testing** | — | ~1 hour | **~$2.12** |

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
