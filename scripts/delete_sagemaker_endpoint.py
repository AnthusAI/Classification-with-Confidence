#!/usr/bin/env python3
"""
Delete a SageMaker endpoint and its configuration to stop charges.
Handles Inference Components used for multi-adapter inference.

Usage:
    python3 scripts/delete_sagemaker_endpoint.py <endpoint-name>
"""

import boto3
import sys
import time

REGION = 'us-east-1'

if len(sys.argv) != 2:
    print("Usage: python3 scripts/delete_sagemaker_endpoint.py <endpoint-name>")
    sys.exit(1)

ENDPOINT_NAME = sys.argv[1]

sagemaker = boto3.client('sagemaker', region_name=REGION)

print(f"🗑️  Deleting SageMaker Endpoint: {ENDPOINT_NAME}")
print(f"   Region: {REGION}")
print("")

# Get endpoint details
try:
    endpoint = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
    config_name = endpoint['EndpointConfigName']
    print(f"   Found endpoint config: {config_name}")
except Exception as e:
    print(f"❌ Error: Could not find endpoint '{ENDPOINT_NAME}'")
    print(f"   {e}")
    sys.exit(1)

# List and delete inference components
print(f"\n1️⃣  Checking for Inference Components...")
try:
    paginator = sagemaker.get_paginator('list_inference_components')
    components = []
    for page in paginator.paginate(EndpointNameEquals=ENDPOINT_NAME):
        components.extend(page['InferenceComponents'])

    if components:
        print(f"   Found {len(components)} component(s) to delete:")
        for comp in components:
            print(f"   - {comp['InferenceComponentName']}")

        # Separate base and adapter components
        base_components = []
        adapter_components = []

        for comp in components:
            comp_name = comp['InferenceComponentName']
            details = sagemaker.describe_inference_component(InferenceComponentName=comp_name)
            if 'BaseInferenceComponentName' in details['Specification']:
                adapter_components.append(comp_name)
            else:
                base_components.append(comp_name)

        # Delete adapter components first (they depend on base components)
        if adapter_components:
            print(f"\n   Deleting {len(adapter_components)} adapter component(s) first...")
            for comp_name in adapter_components:
                print(f"   - Deleting adapter: {comp_name}")
                try:
                    sagemaker.delete_inference_component(InferenceComponentName=comp_name)
                    print(f"     ✅ Deletion started")
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    sys.exit(1)

            # Wait for adapter components to be deleted
            print(f"\n⏳ Waiting for adapter components to be deleted...")
            for comp_name in adapter_components:
                while True:
                    try:
                        resp = sagemaker.describe_inference_component(InferenceComponentName=comp_name)
                        status = resp['InferenceComponentStatus']
                        if status == 'Deleting':
                            time.sleep(5)
                            continue
                    except sagemaker.exceptions.ClientError as e:
                        if 'Could not find' in str(e) or 'ValidationException' in str(e):
                            print(f"   ✅ {comp_name} deleted!")
                            break
                        raise
                    time.sleep(5)

        # Now delete base components
        if base_components:
            print(f"\n   Deleting {len(base_components)} base component(s)...")
            for comp_name in base_components:
                print(f"   - Deleting base: {comp_name}")
                try:
                    sagemaker.delete_inference_component(InferenceComponentName=comp_name)
                    print(f"     ✅ Deletion started")
                except Exception as e:
                    print(f"     ❌ Error: {e}")
                    sys.exit(1)

            # Wait for base components to be deleted
            print(f"\n⏳ Waiting for base components to be deleted...")
            for comp_name in base_components:
                while True:
                    try:
                        resp = sagemaker.describe_inference_component(InferenceComponentName=comp_name)
                        status = resp['InferenceComponentStatus']
                        if status == 'Deleting':
                            time.sleep(5)
                            continue
                    except sagemaker.exceptions.ClientError as e:
                        if 'Could not find' in str(e) or 'ValidationException' in str(e):
                            print(f"   ✅ {comp_name} deleted!")
                            break
                        raise
                    time.sleep(5)
    else:
        print(f"   No Inference Components found")
except Exception as e:
    print(f"   ❌ Error checking components: {e}")
    sys.exit(1)

# Delete endpoint
print(f"\n2️⃣  Deleting endpoint...")
try:
    sagemaker.delete_endpoint(EndpointName=ENDPOINT_NAME)
    print("   ✅ Endpoint deletion started")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Wait for deletion
print(f"\n⏳ Waiting for endpoint deletion...")
while True:
    try:
        resp = sagemaker.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp['EndpointStatus']
        print(f"   Status: {status}")
        if status == 'Deleting':
            time.sleep(5)
            continue
    except sagemaker.exceptions.ClientError as e:
        if 'Could not find' in str(e) or 'ValidationException' in str(e):
            print("   ✅ Endpoint deleted!")
            break
        raise
    time.sleep(5)

# Delete endpoint config
print(f"\n3️⃣  Deleting endpoint config...")
try:
    sagemaker.delete_endpoint_config(EndpointConfigName=config_name)
    print("   ✅ Endpoint config deleted!")
except Exception as e:
    print(f"   ❌ Error: {e}")

print(f"\n✅ Cleanup complete!")
print(f"   Endpoint '{ENDPOINT_NAME}' has been deleted")
print(f"   Charges have stopped")
