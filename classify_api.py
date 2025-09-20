#!/usr/bin/env python3
"""
FastAPI server for sentiment classification with confidence scoring.

This API loads the model once at startup and keeps it in memory for fast responses.

Usage:
    python classify_api.py
    
    # Then use curl to classify text:
    curl -X POST "http://localhost:8000/classify" \
         -H "Content-Type: application/json" \
         -d '{"text": "Your text here"}'

Returns:
    {"prediction": "positive", "confidence": 0.9999}
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from logprobs_confidence import TransformerLogprobsClassifier
import uvicorn
import logging
import os
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Classification API",
    description="Fast sentiment classification with confidence scoring using Llama 3.1 (base and fine-tuned)",
    version="2.0.0"
)

# Global classifier instances
base_classifier = None
finetuned_classifier = None

class TextInput(BaseModel):
    text: str
    model_type: Optional[str] = "base"  # "base" or "finetuned"

class ClassificationResult(BaseModel):
    prediction: str
    confidence: float
    text: str
    model_used: str

@app.on_event("startup")
async def startup_event():
    """Load the models once at startup."""
    global base_classifier, finetuned_classifier
    logger.info("🚀 Starting up FastAPI server...")
    
    # Load base model
    logger.info("📦 Loading base Llama 3.1 model (this may take a few minutes)...")
    try:
        base_classifier = TransformerLogprobsClassifier()
        logger.info("✅ Base model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load base model: {e}")
        raise e
    
    # Try to load fine-tuned model if it exists
    finetuned_model_path = "fine_tuned_sentiment_model"
    if os.path.exists(finetuned_model_path):
        logger.info("🎯 Loading fine-tuned model...")
        try:
            finetuned_classifier = TransformerLogprobsClassifier(model_path=finetuned_model_path)
            logger.info("✅ Fine-tuned model loaded successfully!")
        except Exception as e:
            logger.error(f"⚠️ Failed to load fine-tuned model: {e}")
            logger.info("📝 Fine-tuned model will not be available")
    else:
        logger.info("📝 No fine-tuned model found - only base model available")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Sentiment Classification API",
        "status": "ready" if base_classifier is not None else "loading",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "finetuned_available": finetuned_classifier is not None
    }

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if base_classifier is not None else "unhealthy",
        "base_model_loaded": base_classifier is not None,
        "finetuned_model_loaded": finetuned_classifier is not None,
        "api_version": "2.0.0"
    }

@app.post("/classify", response_model=ClassificationResult)
async def classify_text(input_data: TextInput) -> ClassificationResult:
    """
    Classify text sentiment with confidence score.
    
    Args:
        input_data: JSON with 'text' and optional 'model_type' fields
        
    Returns:
        JSON with prediction, confidence, original text, and model used
    """
    # Select the appropriate classifier
    if input_data.model_type == "finetuned":
        if finetuned_classifier is None:
            raise HTTPException(status_code=400, detail="Fine-tuned model not available")
        classifier = finetuned_classifier
        model_used = "finetuned"
    else:
        if base_classifier is None:
            raise HTTPException(status_code=503, detail="Base model not loaded yet")
        classifier = base_classifier
        model_used = "base"
    
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    try:
        logger.info(f"🔍 Classifying: {input_data.text[:50]}...")
        
        # Get classification result
        result = classifier.get_real_logprobs_confidence(input_data.text)
        
        prediction = result.get('prediction', 'unknown')
        confidence = result.get('confidence', 0.0)
        
        logger.info(f"✅ Result: {prediction} ({confidence:.4f})")
        
        return ClassificationResult(
            prediction=prediction,
            confidence=confidence,
            text=input_data.text,
            model_used=model_used
        )
        
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/classify")
async def classify_get(
    text: str, 
    model_type: str = Query("base", description="Model to use: 'base' or 'finetuned'")
) -> ClassificationResult:
    """
    Classify text via GET request (for simple curl usage).
    
    Args:
        text: Text to classify as query parameter
        model_type: Model to use ("base" or "finetuned")
        
    Returns:
        JSON with prediction, confidence, original text, and model used
    """
    input_data = TextInput(text=text, model_type=model_type)
    return await classify_text(input_data)

if __name__ == "__main__":
    print("🚀 Starting Enhanced Sentiment Classification API Server...")
    print("📝 Usage examples:")
    print("   # Base model:")
    print("   curl 'http://localhost:8000/classify?text=This is amazing!'")
    print("   # Fine-tuned model:")
    print("   curl 'http://localhost:8000/classify?text=This is amazing!&model_type=finetuned'")
    print("   # POST request:")
    print("   curl -X POST 'http://localhost:8000/classify' -H 'Content-Type: application/json' -d '{\"text\": \"This is terrible!\", \"model_type\": \"finetuned\"}'")
    print("")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )


