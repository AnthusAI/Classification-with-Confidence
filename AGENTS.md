# AGENTS.md - Project Implementation Plan

**NOTE**: This document is intended for AI agents working on this project. The `README.md` file is the user-facing documentation that explains the final product to humans who want to understand and use the code.

## Project Overview

This project demonstrates how to extract confidence scores from Large Language Models (LLMs) when used as classifiers. The core insight is that LLMs show higher response consistency on familiar concepts and lower consistency on unfamiliar ones.

## Implementation Strategy

### Core Hypothesis
- **Familiar Language**: LLMs will show high consistency (high confidence) on standard English sentiment expressions
- **Unfamiliar Language**: LLMs will show low consistency (low confidence) on Gen Z/Alpha slang they haven't seen during training
- **Improvement**: Few-shot learning can improve confidence on previously unfamiliar terms

### Technical Approach

#### 1. Confidence Through Consistency
Instead of trying to extract log probabilities (which aren't available in most Ollama models), we measure confidence through:
- **Multiple Queries**: Ask the model the same question 5-10 times
- **Response Agreement**: Higher agreement = higher confidence
- **Consistency Score**: Percentage of responses that agree with the majority prediction

#### 2. Test Dataset Design
- **Standard Examples**: Clear sentiment expressions ("This is amazing!", "I hate this")
- **Gen Z Slang**: Terms likely unknown to older models ("That's so skibidi", "This is bussin fr")
- **Expected Behavior**: High confidence on standard, low confidence on slang

#### 3. Few-Shot Learning Simulation
- **Before/After Comparison**: Test confidence with and without few-shot examples
- **Targeted Examples**: Provide slang-to-sentiment mappings in the prompt
- **Expected Improvement**: Higher confidence on slang after few-shot learning

## Implementation Status

### ✅ Completed Components

1. **`classifier.py`**: Basic Ollama-based sentiment classifier
   - Handles API communication with local Ollama instance
   - Robust error handling and response normalization
   - Support for few-shot examples in prompts

2. **`datasets.py`**: Test data and examples
   - Standard English sentiment examples (high confidence expected)
   - Gen Z/Alpha slang examples (low confidence expected)
   - Few-shot learning examples for improvement
   - Organized by categories for systematic testing

3. **`confidence.py`**: Confidence scoring through consistency
   - Multiple independent classifications
   - Response agreement calculation
   - Batch processing capabilities
   - Evaluation metrics and analysis

4. **`fine_tuning.py`**: Few-shot learning simulation
   - Before/after comparison framework
   - Category-wise analysis of improvement
   - Statistical significance testing
   - Comprehensive reporting

5. **`README.md`**: User-facing documentation
   - Educational article explaining the concept
   - Quick start instructions
   - Expected results and learning outcomes

### 🔄 In Progress

6. **`main.py`**: Main runner script for easy execution

### 📋 Next Steps

7. **Create examples showing confidence differences**: Run actual tests to demonstrate the hypothesis
8. **Add requirements.txt**: List all Python dependencies
9. **Test end-to-end workflow**: Verify everything works together

## Technical Details

### Dependencies
- `requests`: For Ollama API communication
- `statistics`: For confidence score calculations
- Standard library only (no heavy ML frameworks)

### Model Requirements
- Any Ollama-compatible model (e.g., llama3.2, llama3.1)
- Local inference for privacy and consistency
- No special API features required (works with basic text generation)

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

## Expected Experimental Results

### High Confidence Examples
- "This is absolutely amazing!" → positive (confidence: 0.9-1.0)
- "I hate this so much!" → negative (confidence: 0.9-1.0)
- "It's okay, I guess." → neutral (confidence: 0.8-1.0)

### Low Confidence Examples (Pre-Few-Shot)
- "That's so skibidi" → ??? (confidence: 0.2-0.6)
- "This is bussin fr" → ??? (confidence: 0.2-0.6)
- "You're literally sigma" → ??? (confidence: 0.2-0.6)

### Improved Confidence (Post-Few-Shot)
- Same slang terms should show higher confidence after providing examples
- Expected improvement: +0.2 to +0.4 confidence points

## Agent Guidelines

### When Working on This Project

1. **Testing Strategy**: Always test with actual Ollama instance running
2. **Error Handling**: Gracefully handle API failures and invalid responses
3. **Reproducibility**: Use consistent parameters (temperature, sampling) for fair comparison
4. **Documentation**: Keep both technical (AGENTS.md) and user (README.md) docs updated
5. **Validation**: Verify hypothesis through actual experimentation, not just theoretical implementation

### Code Style
- Clear, educational code (this is for learning, not production)
- Extensive comments explaining the confidence methodology
- Modular design for easy experimentation
- Progress indicators for long-running experiments

### Validation Approach
1. **Connection Test**: Verify Ollama is running and model is available
2. **Basic Classification**: Test standard sentiment examples
3. **Confidence Measurement**: Verify consistency-based confidence works
4. **Slang Detection**: Confirm low confidence on unfamiliar terms
5. **Few-Shot Improvement**: Demonstrate confidence improvement

## Success Criteria

The project succeeds if it demonstrates:

1. **Confidence Correlation**: Higher confidence scores correlate with accuracy
2. **Domain Gap Detection**: Lower confidence on unfamiliar Gen Z slang
3. **Few-Shot Improvement**: Measurable confidence increase after few-shot learning
4. **Educational Value**: Clear demonstration of uncertainty quantification in LLMs
5. **Reproducibility**: Anyone can run the code and see similar results

## Future Extensions

- **Calibration**: Convert raw consistency scores to calibrated probabilities
- **Multi-Domain**: Extend beyond sentiment to other classification tasks
- **Different Models**: Compare confidence patterns across different LLMs
- **Real Logprobs**: Integrate with models that provide actual probability distributions

---

*This document guides AI agents in understanding and extending this educational demonstration of LLM confidence estimation.*