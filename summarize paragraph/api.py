import os
import logging
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODELS_TO_TRY = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

def analyze_document(text: str, token: str, models: List[str] = MODELS_TO_TRY) -> str:
    """Analyzes text to extract a summary, action items, and decisions."""
    messages = [
        {"role": "system", "content": "You are a highly efficient assistant. Format your responses clearly with headings."},
        {"role": "user", "content": f"Please analyze the following text and provide:\n1. A brief Summary.\n2. A list of Action Items (who needs to do what).\n3. A list of Key Decisions made.\n\nText to analyze:\n{text}"}
    ]

    for model_id in models:
        logger.info(f"Attempting analysis with model: {model_id}...")
        client = InferenceClient(model=model_id, token=token)
        try:
            response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.3)
            logger.info(f"Success with {model_id}!\n")
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "not supported" in error_msg or "loading" in error_msg:
                logger.warning(f"Model {model_id} unavailable. Trying fallback...")
                continue 
            logger.error(f"Critical API error: {str(e)}")
            return f"Error: {str(e)}"
            
    return "Error: All configured models are currently unavailable."

# --- FastAPI Setup ---
app = FastAPI(title="Document Analysis API")

# Define the expected JSON payload format
class AnalyzeRequest(BaseModel):
    text: str

@app.post("/api/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    # SECURE YOUR TOKEN: Do not hardcode this in production.
    hf_token = os.getenv("HF_TOKEN") or "" 
    
    logger.info("Received analysis request.")
    result = analyze_document(text=request.text, token=hf_token)
    
    if result.startswith("Error:"):
        raise HTTPException(status_code=503, detail=result)
        
    return {"analysis": result}