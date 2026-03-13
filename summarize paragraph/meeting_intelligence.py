import os
import json
import logging
from typing import List, Optional
from huggingface_hub import InferenceClient

from prompt import get_meeting_prompt

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
                continue
            logger.error(f"API error with {model_id}: {str(e)}")
            return f"Error: {str(e)}"
    return None

def analyze_meeting(transcript: str, token: str, models: Optional[List[str]] = None) -> dict:
    """
    Analyzes a meeting transcript to extract summary, tasks, risks, and decision points.
    Returns a dictionary containing the extracted information.
    """
    if models is None:
        models = MODELS_TO_TRY
        
    messages = get_meeting_prompt(transcript)
    result_text = _call_llm(messages, token, models)
    
    if not result_text:
        return {"error": "All configured models are currently unavailable."}
        
    if result_text.startswith("Error:"):
        return {"error": result_text}
        
    # Process the LLM response into JSON
    try:
        clean_res = result_text.strip()
        if clean_res.startswith('```json'):
            clean_res = clean_res[7:-3].strip()
        elif clean_res.startswith('```'):
            clean_res = clean_res[3:-3].strip()
            
        parsed_result = json.loads(clean_res)
        
        # Ensure the required keys exist in the output
        for key in ["summary", "tasks", "risks", "decision_points"]:
            if key not in parsed_result:
                parsed_result[key] = [] if key != "summary" else ""
                
        return parsed_result
        
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response into JSON.")
        return {
            "error": "Failed to parse LLM response into JSON.",
            "raw_response": result_text
        }

if __name__ == "__main__":
    # Test the module directly
    hf_token = os.getenv("HF_TOKEN") or "HF_TOKEN"
    
    sample_transcript = """
    Alice: Let's discuss the new Q4 marketing campaign. Bob, are the creatives ready?
    Bob: Mostly. The banner ads are done, but we're still waiting on the video assets from the agency. 
         They said it might be delayed by a week, which could push our launch date.
    Alice: That's a huge risk. We need to launch before Black Friday. Let's decide right now to decouple the video 
           launch from the banner ads. We'll go live with banners first.
    Bob: Agreed. I'll email the agency today to get a finalized delivery date for the video.
    Charlie: I'll coordinate with the web team to update the landing page for the banners.
    Alice: Perfect. Make sure the landing page is live by next Tuesday.
    """
    
    print("\n" + "="*40)
    print("MEETING INTELLIGENCE TOOL - TEST RUN")
    print("="*40)
    
    result = analyze_meeting(sample_transcript, hf_token)
    
    print("\nEXTRACTED JSON OUTPUT:")
    print(json.dumps(result, indent=2))
    print("="*40 + "\n")
