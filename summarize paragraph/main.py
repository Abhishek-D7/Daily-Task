import os
import json
import logging
from typing import List, Optional
from huggingface_hub import InferenceClient

from prompt import get_analysis_prompt, get_validation_prompt

# Configure logging for professional, clean console output
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# A list of models to try, ordered from most capable/popular to fallbacks
DEFAULT_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct",      # Extremely fast, highly supported right now
    "Qwen/Qwen2.5-7B-Instruct",              # Very reliable, great at instruction following
    "mistralai/Mistral-Nemo-Instruct-2407",  # Newer Mistral model, often supported
    "meta-llama/Meta-Llama-3-8B-Instruct"    # The flagship free model
]

def _call_llm(messages: list[dict], token: str, models: List[str] = DEFAULT_MODELS) -> Optional[str]:
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

def analyze_document(text: str, token: str, models: Optional[List[str]] = None) -> str:
    """Analyzes text to extract information and validates the extraction."""
    if models is None:
        models = DEFAULT_MODELS

    # 1. Generate Initial Analysis
    analysis_messages = get_analysis_prompt(text)
    initial_analysis = _call_llm(analysis_messages, token, models)
    
    if not initial_analysis:
        return "Error: All configured models are currently unavailable on the free tier. Please try again later."
        
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

def main() -> None:
    # Handle credentials (prefers environment variable, falls back to hardcoded string)
    hf_token = os.getenv("HF_TOKEN") or "HF_TOKEN"
    
    sample_text = """
    Meeting notes: Q3 Product Sync - October 12th.
    Sarah kicked off the meeting by reviewing the Q2 metrics. We missed our user retention goal by 2%, 
    mostly due to the bug in the onboarding flow. Mark proposed we shift the engineering team's focus 
    to fixing the onboarding pipeline immediately. We all agreed that this is the top priority for Q3. 
    As a result, the launch of the dark mode feature will be delayed to Q4. 
    Sarah will draft a new project timeline by this Friday. Mark needs to sync with the QA team tomorrow 
    to figure out how the onboarding bug slipped through testing. Finally, we decided to allocate an extra 
    $5,000 to the marketing budget for the upcoming holiday campaign, which Jenny will manage.
    """
    
    logger.info("=== STARTING ANALYSIS ===")
    
    # Pass the text and token explicitly to the function
    result = analyze_document(text=sample_text, token=hf_token)
    
    print("\n=== ANALYSIS RESULTS ===")
    print(result)

if __name__ == "__main__":
    main()
