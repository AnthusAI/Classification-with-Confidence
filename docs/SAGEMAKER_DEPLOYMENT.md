# Deploying to Amazon SageMaker with LoRA Adapters

This guide documents how to deploy this sentiment classification model to Amazon SageMaker using Inference Components with LoRA adapters for multi-adapter inference.

## Overview

**What is Multi-Adapter Inference?**

SageMaker Inference Components allow you to deploy hundreds of LoRA adapters on a single endpoint, enabling cost-efficient serving of customer-specific fine-tuned models. Instead of deploying one endpoint per customer (expensive!), you deploy one base model and dynamically load adapter weights as needed.

Read more: [Easily deploy and manage hundreds of LoRA adapters with SageMaker efficient multi-adapter inference](https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/)

**Architecture:**
```
┌─────────────────────────────────────────────┐
│   Endpoint (ml.g6e.xlarge)                  │
│   4 vCPUs, 16GB RAM, 48GB GPU (NVIDIA L40S) │
│                                             │
│   ┌──────────────────────────────────────┐ │
│   │  Base Component                       │ │
│   │  - Llama 3.1-8B-Instruct             │ │
│   │  - vLLM with LoRA support            │ │
│   │  - 2 CPUs, 4GB RAM, 1 GPU            │ │
│   └──────────────────────────────────────┘ │
│                                             │
│   ┌──────────────────────────────────────┐ │
│   │  Adapter Component: Sentiment         │ │
│   │  - LoRA weights (flat structure)      │ │
│   │  - Loaded from S3                     │ │
│   └──────────────────────────────────────┘ │
│                                             │
│   (Can add 10+ more adapter components)     │
└─────────────────────────────────────────────┘
```

## Prerequisites

### 1. AWS Account Setup

- AWS Account with SageMaker access
- IAM role with permissions: `LlamaSageMakerExecutionRole`
  - `AmazonSageMakerFullAccess`
  - S3 bucket access for model artifacts

### 2. HuggingFace Token

The base model (Llama 3.1-8B-Instruct) is gated on HuggingFace. You need a token with access:

1. Go to https://huggingface.co/settings/tokens
2. Create a token with "Read" permission
3. Accept the Llama license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
4. Set environment variable:
   ```bash
   export HF_TOKEN="hf_..."
   ```

### 3. ml.g6e.xlarge Quota

Request quota increase for `ml.g6e.xlarge for endpoint usage` in us-east-1:

```bash
./scripts/request_g6e_quota.sh
```

Check status:
```bash
./scripts/check_quota_status.sh
```

**Important**: Quota is approved when the value changes from `0.0` to `1.0` (not when status is `CASE_OPENED`).

See [HOW_TO_REQUEST_QUOTA.md](../HOW_TO_REQUEST_QUOTA.md) for detailed instructions.

## Quick Start

### 1. Fine-Tune Your Model Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Fine-tune the model
python fine_tune_model.py

# This creates: fine_tuned_sentiment_model/ with LoRA adapter weights
```

### 2. Package LoRA Adapter for SageMaker

**CRITICAL**: vLLM requires a **flat directory structure** for LoRA adapters. Files must be at the root of the tarball, not in a subdirectory.

```bash
cd fine_tuned_sentiment_model
tar -czf ../sentiment_adapter.tar.gz adapter_model.safetensors adapter_config.json
cd ..
```

**Correct structure:**
```
sentiment_adapter.tar.gz
├── adapter_model.safetensors
└── adapter_config.json
```

**WRONG structure (will fail):**
```
sentiment_adapter.tar.gz
└── sentiment/
    ├── adapter_model.safetensors
    └── adapter_config.json
```

### 3. Upload to S3

```bash
aws s3 cp sentiment_adapter.tar.gz s3://your-bucket/adapters/
```

### 4. Deploy to SageMaker

Update S3 paths in `scripts/deploy_sagemaker.py`:

```python
ADAPTER_S3 = "s3://your-bucket/adapters/sentiment_adapter.tar.gz"
```

Deploy:

```bash
export HF_TOKEN="hf_..."
python3 scripts/deploy_sagemaker.py
```

This script will:
1. Create endpoint (5-10 minutes)
2. Create base component with Llama 3.1-8B (10-15 minutes)
3. Create adapter component (2-5 minutes)
4. Test both components
5. Return endpoint names for invocation

**Total deployment time: ~20-30 minutes**

## Instance Configuration

### Tested Configuration (WORKING)

**Instance**: `ml.g6e.xlarge`
- 4 vCPUs
- 16GB System RAM
- 48GB GPU Memory (NVIDIA L40S)
- **Cost**: ~$1.25/hour

**Base Component Resources**:
```python
'ComputeResourceRequirements': {
    'NumberOfCpuCoresRequired': 2,        # NOT 4!
    'NumberOfAcceleratorDevicesRequired': 1,
    'MinMemoryRequiredInMb': 4096,        # NOT 12GB!
}
```

**vLLM Configuration**:
```python
'Environment': {
    'HF_MODEL_ID': 'meta-llama/Llama-3.1-8B-Instruct',
    'HUGGING_FACE_HUB_TOKEN': HF_TOKEN,
    'OPTION_ROLLING_BATCH': 'vllm',
    'OPTION_ENABLE_LORA': 'true',
    'OPTION_MAX_LORAS': '10',
    'OPTION_MAX_LORA_RANK': '64',
    'OPTION_MAX_MODEL_LEN': '4096',
    'OPTION_GPU_MEMORY_UTILIZATION': '0.8',
}
```

**RuntimeConfig**: REQUIRED for base component
```python
RuntimeConfig={'CopyCount': 1}
```

### Why NOT 4 CPUs / 12GB RAM?

Inference Components architecture requires headroom for SageMaker orchestration. Requesting all available resources causes allocation errors:

- Instance: 4 vCPUs total
- SageMaker needs: ~2 vCPUs for orchestration
- Component gets: 2 vCPUs

Same for RAM:
- Instance: 16GB total
- SageMaker needs: ~12GB for orchestration
- Component gets: 4GB

## Invoking the Endpoint

### Base Model (No Adapter)

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    InferenceComponentName='your-base-component-name',
    Body=json.dumps({
        "inputs": "Classify the sentiment: 'This movie was great!'",
        "parameters": {"max_new_tokens": 50}
    }),
    ContentType="application/json"
)

result = json.loads(response['Body'].read().decode())
print(result)
```

### LoRA Adapter Model

```python
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    InferenceComponentName='your-adapter-component-name',  # Use adapter component!
    Body=json.dumps({
        "inputs": "Classify the sentiment: 'This movie was great!'",
        "parameters": {"max_new_tokens": 50}
    }),
    ContentType="application/json"
)

result = json.loads(response['Body'].read().decode())
print(result)
```

## Cost Analysis

### Development/Testing (2 hours/day)
- ml.g6e.xlarge: ~$1.25/hour
- Daily: $2.50
- Monthly: $75
- Annual: $912

### Production (24/7)
- ml.g6e.xlarge: ~$1.25/hour
- Daily: $30
- Monthly: $912
- Annual: $10,950

**Cost Savings vs Traditional Approach**:
- Traditional: 10 customers × 10 endpoints × $1.25/hour = $12.50/hour
- Multi-Adapter: 1 endpoint × $1.25/hour = $1.25/hour
- **Savings**: 90% ($11.25/hour, $8,212/month)

## Cleanup

Delete endpoint when done to avoid ongoing charges:

```bash
aws sagemaker delete-endpoint --endpoint-name your-endpoint-name --region us-east-1
```

The script will also print the delete command at the end.

## Troubleshooting

### "Not enough hardware resources"

**Cause**: Requesting too many CPUs/RAM for the component.

**Solution**: Use 2 CPUs and 4GB RAM (not 4 CPUs / 12GB RAM).

### "Loading lora ... failed"

**Cause**: LoRA adapter tarball has nested directory structure.

**Solution**: Repackage adapter with files at root level (flat structure).

### "ResourceLimitExceeded"

**Cause**: ml.g6e.xlarge quota not approved yet.

**Solution**: Check quota status and wait for approval.

### Console shows "MissingRequiredParameter - Missing required key 'ModelName'"

**Not an error!** This is expected for Inference Components endpoints. You attach the model to the component, not directly to the endpoint.

## Key Learnings

1. **Flat Structure Required**: vLLM expects adapter files at tarball root, not in subdirectories
2. **Resource Overhead**: Inference Components need headroom - don't request all instance resources
3. **RuntimeConfig Required**: Base components need `RuntimeConfig={'CopyCount': 1}`
4. **Adapter Components**: Don't include `VariantName` or `RuntimeConfig` for adapter components
5. **Quota Approval**: `CASE_OPENED` status means pending, not approved - wait for quota value > 0

## Next Steps

- Add more LoRA adapters for different customers/use-cases
- Implement tiered caching strategy (hot/warm/cold adapters)
- Set up CloudWatch monitoring and alarms
- Configure auto-scaling policies
- Implement adapter versioning and rollback

## References

- [AWS Blog: Multi-Adapter Inference](https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/)
- [SageMaker Inference Components Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-components.html)
- [LMI Container Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
- [vLLM LoRA Support](https://docs.vllm.ai/en/latest/models/lora.html)
## Internal Documentation

- [GPU_MEMORY_ANALYSIS.txt](../GPU_MEMORY_ANALYSIS.txt) - Why G5 instances don't work
- [SAGEMAKER_LESSONS_LEARNED.md](../SAGEMAKER_LESSONS_LEARNED.md) - Deployment insights and gotchas
- [SAGEMAKER_G6E_QUOTA_STATUS.md](../SAGEMAKER_G6E_QUOTA_STATUS.md) - Quota request timeline and status

