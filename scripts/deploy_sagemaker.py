import boto3
import time
import json
import os

REGION = 'us-east-1'
ENDPOINT_NAME = f"llama-g6e-{int(time.time())}"
CONFIG_NAME = f"{ENDPOINT_NAME}-config"
BASE_IC_NAME = f"{ENDPOINT_NAME}-base"
ADAPTER_IC_NAME = f"{ENDPOINT_NAME}-adapter"

# Use LMI container optimized for LoRA
IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.31.0-lmi13.0.0-cu124"
INSTANCE_TYPE = "ml.g6e.xlarge"  # 48GB GPU, NVIDIA L40S

HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_S3 = "s3://llama-sagemaker-models-1763729683/adapters/sentiment_correct.tar.gz"

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    print("❌ Error: HF_TOKEN not set. Please set the HuggingFace Hub token as an environment variable.")
    exit(1)

account_id = boto3.client('sts').get_caller_identity()['Account']
role_arn = f"arn:aws:iam::{account_id}:role/LlamaSageMakerExecutionRole"

sagemaker = boto3.client('sagemaker', region_name=REGION)
runtime = boto3.client('sagemaker-runtime', region_name=REGION)

print(f"🚀 Deploying with ml.g6e.xlarge (48GB GPU)!")
print(f"   Endpoint: {ENDPOINT_NAME}")
print(f"   Instance: {INSTANCE_TYPE} - NVIDIA L40S with 48GB memory")
print(f"   Container: LMI with vLLM + LoRA support")
print("")

# 1. Create Endpoint Config
print(f"📝 Creating Endpoint Config...")
try:
    sagemaker.create_endpoint_config(
        EndpointConfigName=CONFIG_NAME,
        ExecutionRoleArn=role_arn,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'InstanceType': INSTANCE_TYPE,
            'InitialInstanceCount': 1,
        }]
    )
    print(f"   ✅ Config created")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# 2. Create Endpoint
print(f"\n🌐 Creating Endpoint...")
try:
    sagemaker.create_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=CONFIG_NAME
    )
    print(f"   ✅ Endpoint creation started")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Wait for endpoint
print(f"\n⏳ Waiting for Endpoint (~5-10 minutes)...")
iteration = 0
while True:
    resp = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
    status = resp['EndpointStatus']
    iteration += 1
    if iteration % 6 == 0:
        print(f"   [{iteration//2} min] Status: {status}")
    if status == 'InService':
        print("   ✅ Endpoint InService!")
        break
    elif status == 'Failed':
        print(f"   ❌ Failed: {resp.get('FailureReason')}")
        exit(1)
    time.sleep(10)

# 3. Create Base Component with optimized settings for 48GB GPU
print(f"\n📦 Creating Base LMI Component...")
try:
    sagemaker.create_inference_component(
        InferenceComponentName=BASE_IC_NAME,
        EndpointName=ENDPOINT_NAME,
        VariantName='AllTraffic',
        Specification={
            'Container': {
                'Image': IMAGE,
                'Environment': {
                    'HF_MODEL_ID': HF_MODEL_ID,
                    'HUGGING_FACE_HUB_TOKEN': HF_TOKEN,
                    'OPTION_ROLLING_BATCH': 'vllm',
                    'OPTION_ENABLE_LORA': 'true',
                    'OPTION_MAX_LORAS': '10',
                    'OPTION_MAX_LORA_RANK': '64',
                    'OPTION_MAX_MODEL_LEN': '4096',  # Conservative for xlarge
                    'OPTION_GPU_MEMORY_UTILIZATION': '0.8',  # Conservative for stability
                }
            },
            'ComputeResourceRequirements': {
                'NumberOfCpuCoresRequired': 2,        # Sized for ml.g6e.xlarge, leaving headroom for the SageMaker agent.
                'NumberOfAcceleratorDevicesRequired': 1,
                'MinMemoryRequiredInMb': 4096,        # Sized for ml.g6e.xlarge, leaving headroom for the SageMaker agent.
            }
        }
        # Note: Do not include RuntimeConfig for adapter components!
        # Base components require a CopyCount to scale the underlying hardware.
        RuntimeConfig={'CopyCount': 1}
    )
    print("   ✅ Request sent!")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Wait for Base
print("\n⏳ Waiting for Base Component (~10-15 minutes)...")
iteration = 0
while True:
    resp = sagemaker.describe_inference_component(InferenceComponentName=BASE_IC_NAME)
    status = resp['InferenceComponentStatus']
    iteration += 1
    if iteration % 6 == 0:
        print(f"   [{iteration//2} min] Status: {status}")

    if status == 'InService':
        print("   ✅ Base InService!")
        break
    elif status == 'Failed':
        reason = resp.get('FailureReason', 'Unknown')
        print(f"   ❌ Failed: {reason}")
        exit(1)
    time.sleep(10)

# 4. Create Adapter
print(f"\n🎯 Creating LoRA Adapter Component...")
try:
    sagemaker.create_inference_component(
        InferenceComponentName=ADAPTER_IC_NAME,
        EndpointName=ENDPOINT_NAME,
        Specification={
            'BaseInferenceComponentName': BASE_IC_NAME,
            'Container': {
                'ArtifactUrl': ADAPTER_S3
            },
        },
    )
    print("   ✅ Request sent!")
except Exception as e:
    print(f"   ❌ Error: {e}")
    exit(1)

# Wait for Adapter
print("\n⏳ Waiting for Adapter Component...")
while True:
    resp = sagemaker.describe_inference_component(InferenceComponentName=ADAPTER_IC_NAME)
    status = resp['InferenceComponentStatus']
    print(f"   Status: {status}")
    if status == 'InService':
        print("   ✅ Adapter InService!")
        break
    elif status == 'Failed':
        print(f"   ❌ Failed: {resp.get('FailureReason')}")
        exit(1)
    time.sleep(10)

# 5. Test Both
print("\n🧪 Testing Both Components...")
test_texts = [
    "This movie was absolutely terrible!",
    "Best film I've ever seen!",
    "It was okay, nothing special."
]

for text in test_texts:
    print(f"\n{'='*60}")
    print(f"Input: '{text}'")
    print(f"{'='*60}")

    print("\n1. Base Model:")
    try:
        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            InferenceComponentName=BASE_IC_NAME,
            Body=json.dumps({"inputs": text, "parameters": {"max_new_tokens": 30}}),
            ContentType="application/json"
        )
        result = json.loads(resp['Body'].read().decode())
        print(f"   {result}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

    print("\n2. LoRA Fine-tuned Model:")
    try:
        resp = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            InferenceComponentName=ADAPTER_IC_NAME,
            Body=json.dumps({"inputs": text, "parameters": {"max_new_tokens": 30, "adapter_id": "sentiment"}}),
            ContentType="application/json"
        )
        result = json.loads(resp['Body'].read().decode())
        print(f"   {result}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n{'='*60}")
print(f"🎉 DEPLOYMENT SUCCESSFUL!")
print(f"{'='*60}")
print(f"   Endpoint: {ENDPOINT_NAME}")
print(f"   Base Component: {BASE_IC_NAME}")
print(f"   Adapter Component: {ADAPTER_IC_NAME}")
print(f"   Instance: {INSTANCE_TYPE} (48GB GPU)")
print(f"\n💰 Cost: ~$1.25/hour for {INSTANCE_TYPE}")
print(f"💡 Delete when done: aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME} --region {REGION}")
