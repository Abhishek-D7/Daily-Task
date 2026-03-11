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
        {
            "role": "system", 
            "content": (
                "You are a strict data extraction system. Your output must be ONLY a valid, parseable JSON object. "
                "Do not include markdown formatting (like ```json), backticks, or conversational filler. "
                "You are bound by a strict rule: You may ONLY use information explicitly stated in the provided text."
            )
        },
        {
            "role": "user", 
            "content": (
                "Extract the following information from the text below. \n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. If the information for a field is present, extract it concisely.\n"
                "2. If the information is NOT explicitly mentioned in the text, you MUST output exactly \"Not in context\" for string values, or [\"Not in context\"] for lists.\n"
                "3. Do not infer, guess, or use outside knowledge.\n\n"
                "REQUIRED JSON SCHEMA:\n"
                "{\n"
                '  "summary": "String. A 1-2 sentence summary of the text.",\n'
                '  "action_items": ["List of strings. Specific tasks assigned to people or teams."],\n'
                '  "risks": ["List of strings. Potential problems, delays, or threats mentioned."],\n'
                '  "priority_tasks": ["List of strings. Tasks explicitly noted as urgent, immediate, or top priority."]\n'
                "}\n\n"
                f"TEXT TO ANALYZE:\n{text}"
            )
        }
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
    hf_token = os.getenv("HF_TOKEN") or "HF_TOKEN" 
    
    logger.info("Received analysis request.")
    result = analyze_document(text=request.text, token=hf_token)
    
    if result.startswith("Error:"):
        raise HTTPException(status_code=503, detail=result)
        
    return {"analysis": result}