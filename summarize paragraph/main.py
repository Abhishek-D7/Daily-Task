import os
import logging
from typing import List, Optional
from huggingface_hub import InferenceClient

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

def analyze_document(text: str, token: str, models: Optional[List[str]] = None) -> str:
    """
    Analyzes text to extract a summary, action items, and decisions.
    Includes a fallback mechanism to try multiple models if one is unavailable.
    """
    if models is None:
        models = DEFAULT_MODELS

    messages = [
        {
            "role": "system", 
            "content": "You are a highly efficient assistant. Format your responses clearly with headings."
        },
        {
            "role": "user", 
            "content": (
                "Please analyze the following text and provide:\n"
                "1. A brief Summary.\n"
                "2. A list of Action Items (who needs to do what).\n"
                "3. A list of Key Decisions made.\n\n"
                f"Text to analyze:\n{text}"
            )
        }
    ]
    
    # Loop through our list of models
    for model_id in models:
        logger.info(f"Attempting to use model: {model_id}...")
        client = InferenceClient(model=model_id, token=token)
        
        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=512,  
                temperature=0.3
            )
            logger.info(f"Success with {model_id}!")
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = str(e)
            # If it's a model support/loading error, we continue to the next model
            if "not supported" in error_msg.lower() or "loading" in error_msg.lower():
                logger.warning(f"Model unavailable right now. Trying the next one...")
                continue 
            else:
                # If it's an authentication error (bad token), we should stop and report it
                logger.error(f"Critical error occurred: {error_msg}")
                return f"A critical error occurred: {error_msg}"
                
    # If the loop finishes and all models failed
    error_text = "Error: All attempted models are currently unavailable on the free tier. Please try again later."
    logger.error(error_text)
    return error_text

def main() -> None:
    # Handle credentials (prefers environment variable, falls back to hardcoded string)
    hf_token = os.getenv("HF_TOKEN") or ""
    
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