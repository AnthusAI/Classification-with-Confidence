#!/bin/bash
# Fast classification using the API server
# Usage: ./classify_fast.sh "Your text here" [model_type]
# model_type: "base" (default) or "finetuned"

if [ $# -eq 0 ]; then
    echo "Usage: $0 \"Your text to classify\" [model_type]"
    echo "  model_type: 'base' (default) or 'finetuned'"
    echo "Example: $0 \"This is absolutely amazing!\" base"
    echo "Example: $0 \"This is absolutely amazing!\" finetuned"
    exit 1
fi

TEXT="$1"
MODEL_TYPE="${2:-base}"  # Default to "base" if not specified
API_URL="http://localhost:8000"

# Use POST request to avoid URL encoding issues
curl -s -X POST "$API_URL/classify" \
     -H "Content-Type: application/json" \
     -d "{\"text\": $(echo "$TEXT" | python3 -c "import json, sys; print(json.dumps(sys.stdin.read().strip()))"), \"model_type\": \"$MODEL_TYPE\"}" | \
python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    print(f\"{data['prediction']} {data['confidence']} ({data['model_used']})\")
except Exception as e:
    print('error N/A')
"
