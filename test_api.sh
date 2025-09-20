#!/bin/bash
# Simple script to test the classification API

API_URL="http://localhost:8000"

echo "=== Testing Sentiment Classification API ==="
echo ""

# Check if server is running
echo "1. Health check:"
curl -s "$API_URL/health" | python -m json.tool
echo ""

# Test classification via GET
echo "2. Test classification (GET):"
curl -s "$API_URL/classify?text=This%20is%20absolutely%20amazing!" | python -m json.tool
echo ""

# Test classification via POST
echo "3. Test classification (POST):"
curl -s -X POST "$API_URL/classify" \
     -H "Content-Type: application/json" \
     -d '{"text": "This project has been a complete disaster"}' | python -m json.tool
echo ""

echo "=== API Test Complete ==="



