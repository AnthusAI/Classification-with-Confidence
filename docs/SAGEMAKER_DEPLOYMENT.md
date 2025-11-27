# Deploying to Amazon SageMaker with LoRA Adapters

This guide shows how to deploy fine-tuned sentiment classification models to Amazon SageMaker using Inference Components with LoRA adapters for cost-efficient multi-adapter inference.

## Why SageMaker Inference Components?

**The Problem**: If you need to serve fine-tuned models for many customers, the traditional approach of deploying one endpoint per customer becomes prohibitively expensive.

**The Solution**: SageMaker Inference Components allow you to deploy hundreds of LoRA adapters on a single endpoint. You deploy one base model and dynamically load adapter weights as needed, achieving massive cost savings.

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

**Cost Savings**:
- Traditional: 10 customers × $1.25/hour = $12.50/hour ($9,120/month)
- Multi-Adapter: 1 endpoint × $1.25/hour = $1.25/hour ($912/month)
- **Savings**: 90% ($8,208/month for 10 customers)

## Prerequisites

### 1. AWS Account Setup

- AWS Account with SageMaker access
- IAM role: `LlamaSageMakerExecutionRole` with:
  - `AmazonSageMakerFullAccess`
  - S3 bucket access for model artifacts

### 2. HuggingFace Token

The base model (Llama 3.1-8B-Instruct) is gated. You need a token:

1. Go to https://huggingface.co/settings/tokens
2. Create a token with "Read" permission
3. Accept the Llama license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
4. Set environment variable:
   ```bash
   export HF_TOKEN="hf_..."
   ```

## Deployment Steps

### 1. Fine-Tune Your Model Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Fine-tune the model (10-20 minutes on GPU)
python fine_tune_model.py
```

This creates `fine_tuned_sentiment_model/` with LoRA adapter weights.

### 2. Package LoRA Adapter

**Golden Path Requirement**: The vLLM engine used by SageMaker's LMI container requires a **flat directory structure** for the LoRA adapter. The files must be at the root of the tarball.

```bash
# Navigate into the directory with the adapter files
cd fine_tuned_sentiment_model

# Create the tarball with a flat structure
tar -czf ../sentiment_adapter.tar.gz adapter_model.safetensors adapter_config.json
cd ..
```

### 3. Upload to S3

```bash
aws s3 cp sentiment_adapter.tar.gz s3://your-bucket/adapters/
```

### 4. Deploy to SageMaker

Update S3 path in `scripts/deploy_sagemaker.py`:

```python
ADAPTER_S3 = "s3://your-bucket/adapters/sentiment_adapter.tar.gz"
```

Deploy:

```bash
export HF_TOKEN="hf_..."
python3 scripts/deploy_sagemaker.py
```

The script will:
1. Create endpoint (5-10 minutes)
2. Create base component with Llama 3.1-8B (10-15 minutes)
3. Create adapter component (2-5 minutes)
4. Test both components

**Total deployment time: ~20-30 minutes**

## Configuration Details

The `scripts/deploy_sagemaker.py` script handles all of the following configurations for you.

### Instance

-   **Type**: `ml.g6e.xlarge`
-   **GPU**: 48GB NVIDIA L40S
-   **Why**: This instance provides the necessary 48GB of GPU memory for comfortably serving the Llama 3.1-8B model with the vLLM engine and LoRA adapters.

### LMI Container & vLLM Environment

The script uses the AWS Large Model Inference (LMI) container, which is required for the multi-adapter feature. Key environment variables are set to enable LoRA with vLLM:

```python
'Environment': {
    'HF_MODEL_ID': 'meta-llama/Llama-3.1-8B-Instruct',
    'HUGGING_FACE_HUB_TOKEN': HF_TOKEN,
    'OPTION_ROLLING_BATCH': 'vllm',
    'OPTION_ENABLE_LORA': 'true',
    # ... other settings ...
}
```

### Component Resources

-   **CPU & RAM**: The script correctly allocates only a portion of the instance's CPU and RAM (2 cores, 4GB) to the component. This is a key requirement, as SageMaker's agent needs the remaining resources for orchestration.
-   **RuntimeConfig**: The base component is configured with `RuntimeConfig={'CopyCount': 1}` to provision the underlying hardware. This is not needed for adapter components.

## Using the Endpoint

### Base Model (No Adapter)

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')

response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    InferenceComponentName='your-base-component-name',
    Body=json.dumps({
        "inputs": "This movie was amazing!",
        "parameters": {"max_new_tokens": 50}
    }),
    ContentType="application/json"
)

result = json.loads(response['Body'].read().decode())
print(result)
```

### Fine-Tuned Adapter Model

```python
response = runtime.invoke_endpoint(
    EndpointName='your-endpoint-name',
    InferenceComponentName='your-adapter-component-name',
    Body=json.dumps({
        "inputs": "This movie was amazing!",
        "parameters": {"max_new_tokens": 50}
    }),
    ContentType="application/json"
)

result = json.loads(response['Body'].read().decode())
print(result)
```

## Cost Estimates

### Development/Testing (2 hours/day)
- Daily: $2.50
- Monthly: $75
- Annual: $912

### Production (24/7)
- Daily: $30
- Monthly: $912
- Annual: $10,950

### Multi-Customer Scenarios
For 10 customers with traditional deployment:
- Traditional: 10 endpoints × $912/month = $9,120/month
- Multi-Adapter: 1 endpoint × $912/month = $912/month
- **Savings**: $8,208/month (90%)

## Cleanup

Delete endpoint when done:

```bash
aws sagemaker delete-endpoint --endpoint-name your-endpoint-name --region us-east-1
```

## Adding More Adapters

To serve multiple customer-specific adapters, simply repeat the packaging and uploading steps for each new adapter, then create a new Inference Component for each, pointing to its S3 artifact. All adapters will share the same base model, making this a highly scalable and cost-effective solution.

## Next Steps

-   Implement a tiered caching strategy for hot/warm/cold adapters.
-   Set up CloudWatch monitoring and alarms for latency and errors.
-   Configure auto-scaling policies based on request volume.
-   Implement adapter versioning and rollback strategies.

## References

-   [AWS Blog: Multi-Adapter Inference](https://aws.amazon.com/blogs/machine-learning/easily-deploy-and-manage-hundreds-of-lora-adapters-with-sagemaker-efficient-multi-adapter-inference/)
-   [SageMaker Inference Components Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/inference-components.html)
-   [LMI Container Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)
-   [vLLM LoRA Support](https://docs.vllm.ai/en/latest/models/lora.html)
