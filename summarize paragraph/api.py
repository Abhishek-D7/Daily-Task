import os
import json
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient

from prompt import get_analysis_prompt, get_validation_prompt, get_evaluation_prompt, get_qa_grounding_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODELS_TO_TRY = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Meta-Llama-3-8B-Instruct"
]

def _call_llm(messages: list[dict], token: str, models: List[str] = MODELS_TO_TRY) -> Optional[str]:
    """Helper method to iterate through models and call the API."""
    for model_id in models:
        logger.info(f"Attempting API call with model: {model_id}...")
        client = InferenceClient(model=model_id, token=token)
        try:
            response = client.chat_completion(messages=messages, max_tokens=512, temperature=0.3)
            logger.info(f"Success with {model_id}!")
            return response.choices[0].message.content.strip()
        except Exception as e:
            error_msg = str(e).lower()
            if "not supported" in error_msg or "loading" in error_msg:
                logger.warning(f"Model {model_id} unavailable. Trying fallback...")
                continue 
            logger.error(f"API error with {model_id}: {str(e)}")
            return f"Error: {str(e)}"
            
    return None

def analyze_document(text: str, token: str, models: List[str] = MODELS_TO_TRY) -> str:
    """Analyzes text to extract information and validates the extraction."""
    # 1. Generate Initial Analysis
    analysis_messages = get_analysis_prompt(text)
    initial_analysis = _call_llm(analysis_messages, token, models)
    
    if not initial_analysis:
        return "Error: All configured models are currently unavailable."
        
    if initial_analysis.startswith("Error:"):
        return initial_analysis

    # 2. Validate the Analysis
    logger.info("Starting validation of the generated analysis...")
    validation_messages = get_validation_prompt(text, initial_analysis)
    validation_response = _call_llm(validation_messages, token, models)

    if not validation_response or validation_response.startswith("Error:"):
        logger.warning("Validation step failed. Returning unvalidated initial analysis.")
        return initial_analysis

    # 3. Process Validation Response
    try:
        # Clean any possible markdown wrappers around JSON
        clean_val = validation_response.strip()
        if clean_val.startswith('```json'):
            clean_val = clean_val[7:-3].strip()
        elif clean_val.startswith('```'):
            clean_val = clean_val[3:-3].strip()

        val_data = json.loads(clean_val)
        is_valid = val_data.get("is_valid", False)
        reason = val_data.get("auditor_reasoning", "No reason provided")
        
        logger.info(f"Validation result: Valid={is_valid}, Reason={reason}")

        if is_valid:
            return initial_analysis
        
        corrected_json = val_data.get("corrected_json")
        if corrected_json:
            logger.info("Returning corrected JSON from validator.")
            if isinstance(corrected_json, str):
                return corrected_json
            return json.dumps(corrected_json, indent=2)

    except json.JSONDecodeError:
        logger.warning("Failed to parse validator JSON response. Returning initial analysis.")

    return initial_analysis

# --- FastAPI Setup ---
app = FastAPI(title="Document Analysis API")

# Define the expected JSON payload format
class AnalyzeRequest(BaseModel):
    text: str

class EvaluateRequest(BaseModel):
    expected_output: str
    actual_output: str

class QARequest(BaseModel):
    text: str
    query: str


@app.post("/api/analyze")
def analyze_endpoint(request: AnalyzeRequest):
    # SECURE YOUR TOKEN: Do not hardcode this in production.
    hf_token = os.getenv("HF_TOKEN") or "HF_TOKEN" 
    
    logger.info("Received analysis request.")
    result = analyze_document(text=request.text, token=hf_token)
    
    if result.startswith("Error:"):
        raise HTTPException(status_code=503, detail=result)
        
    # Attempt to safely return actual JSON instead of a string-encoded JSON if applicable
    try:
        parsed_result = json.loads(result)
        return {"analysis": parsed_result}
    except json.JSONDecodeError:
        return {"analysis": result}

@app.post("/api/evaluate")
def evaluate_endpoint(request: EvaluateRequest):
    # SECURE YOUR TOKEN: Do not hardcode this in production.
    hf_token = os.getenv("HF_TOKEN") or "HF_TOKEN" 
    
    logger.info("Received evaluation request.")
    messages = get_evaluation_prompt(request.expected_output, request.actual_output)
    
    # We use the same _call_llm helper
    result = _call_llm(messages, hf_token)
    
    if not result:
        raise HTTPException(status_code=503, detail="Error: All configured models are currently unavailable.")
        
    if result.startswith("Error:"):
        raise HTTPException(status_code=503, detail=result)
        
    try:
        # Clean any markdown wrapper
        clean_res = result.strip()
        if clean_res.startswith('```json'):
            clean_res = clean_res[7:-3].strip()
        elif clean_res.startswith('```'):
            clean_res = clean_res[3:-3].strip()
            
        parsed_result = json.loads(clean_res)
        return {"evaluation": parsed_result}
    except json.JSONDecodeError:
        return {"evaluation": {"is_match": False, "reasoning": f"Failed to parse LLM evaluation: {result}"}}

@app.post("/api/qa")
def qa_endpoint(request: QARequest):
    hf_token = os.getenv("HF_TOKEN") or "HF_TOKEN" 
    
    logger.info(f"Received Q&A request for query: {request.query}")
    messages = get_qa_grounding_prompt(request.text, request.query)
    
    result = _call_llm(messages, hf_token)
    
    if not result:
        raise HTTPException(status_code=503, detail="Error: All configured models are currently unavailable.")
        
    if result.startswith("Error:"):
        raise HTTPException(status_code=503, detail=result)
        
    try:
        clean_res = result.strip()
        if clean_res.startswith('```json'):
            clean_res = clean_res[7:-3].strip()
        elif clean_res.startswith('```'):
            clean_res = clean_res[3:-3].strip()
            
        parsed_result = json.loads(clean_res)
        return {"qa_result": parsed_result}
    except json.JSONDecodeError:
        return {"qa_result": {"query": request.query, "answer": "Error parsing LLM response", "confidence_reasoning": result}}
