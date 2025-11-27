#!/usr/bin/env python3
"""
Check SageMaker quota status for ml.g6e instance types.

Usage:
    python3 scripts/check_sagemaker_quota.py
"""

import boto3
import sys

REGION = 'us-east-1'

# ml.g6e instance types we care about
G6E_INSTANCES = [
    'ml.g6e.xlarge',
    'ml.g6e.2xlarge',
    'ml.g6e.4xlarge',
    'ml.g6e.8xlarge',
    'ml.g6e.12xlarge',
    'ml.g6e.16xlarge',
    'ml.g6e.24xlarge',
    'ml.g6e.48xlarge',
]

print(f"🔍 Checking SageMaker Quotas for ml.g6e Instances")
print(f"   Region: {REGION}")
print()

service_quotas = boto3.client('service-quotas', region_name=REGION)

# Service code for SageMaker
SERVICE_CODE = 'sagemaker'

print("📊 Current Quotas:\n")
print(f"{'Instance Type':<20} {'Quota Name':<50} {'Current Value':<15} {'Status'}")
print("=" * 110)

found_any = False

try:
    # List all SageMaker quotas
    paginator = service_quotas.get_paginator('list_service_quotas')

    for page in paginator.paginate(ServiceCode=SERVICE_CODE):
        for quota in page['Quotas']:
            quota_name = quota['QuotaName']

            # Check if this quota is for endpoint usage of any g6e instance
            if 'endpoint usage' in quota_name.lower():
                for instance_type in G6E_INSTANCES:
                    if instance_type in quota_name:
                        found_any = True
                        value = quota['Value']

                        # Determine status
                        if value == 0.0:
                            status = "❌ NOT APPROVED"
                        else:
                            status = f"✅ APPROVED ({int(value)} instance{'s' if value > 1 else ''})"

                        # Truncate quota name if too long
                        display_name = quota_name if len(quota_name) <= 50 else quota_name[:47] + "..."

                        print(f"{instance_type:<20} {display_name:<50} {value:<15.1f} {status}")
                        break

    if not found_any:
        print("\n⚠️  No ml.g6e quotas found. This could mean:")
        print("   - The quotas haven't been requested yet")
        print("   - The region doesn't support ml.g6e instances")
        print("   - There's an API access issue")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print("\nTroubleshooting:")
    print("   - Ensure AWS credentials are configured")
    print("   - Check IAM permissions for service-quotas:ListServiceQuotas")
    sys.exit(1)

print("\n" + "=" * 110)
print("\n💡 Note:")
print("   - Value 0.0 = Quota not approved yet")
print("   - Value > 0 = Number of instances you can run simultaneously")
print("   - For endpoint usage: Each running endpoint counts toward the quota")
print("\n📝 To request a quota increase:")
print("   - AWS Console → Service Quotas → AWS Services → Amazon SageMaker")
print("   - Or use: aws service-quotas request-service-quota-increase")
