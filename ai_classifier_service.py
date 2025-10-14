from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, List, Optional
import json
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. PYDANTIC DATA MODELS ---
# These models enforce the structure of the data going into and coming out of the service.

class ClassificationRequest(BaseModel):
    """Defines the structure for the user input."""
    answer: str = Field(..., description="The user's learning objective or answer to be classified.")

class ClassificationResult(BaseModel):
    """
    Defines the required structured output from the Gemini model,
    where a definitive category is always chosen.
    """
    primaryDomain: str = Field(..., description="The final, most suitable category for the user's response. This can be one of the existing categories or a newly generated one if no existing category fits.")
    confidenceScore: float = Field(..., ge=0.0, le=1.0, description="The classification confidence (0.0 to 1.0).")
    justification: str = Field(..., description="A concise, single-sentence explanation for the chosen classification.")

# --- 2. CLASSIFICATION CONFIGURATION ---

BASE_SYSTEM_PROMPT = """
You are an intelligent engine for real-time survey analysis. Carefully read and interpret each user response, then assign it to the most appropriate category based on the survey question and existing categories.

**Instructions:**
1. Evaluate the user's answer in the context of the survey question.
2. First, check if the answer reasonably fits any of the **Existing Categories** (semantic meaning counts, not just exact wording). 
3. If it fits an existing category, set `primaryDomain` to that exact category name. Always prefer existing categories over creating new ones.
4. Only if the answer does not reasonably fit any existing category, create a new, concise category name (distinct from the user's answer) and assign it to `primaryDomain`.
5. Always follow the provided JSON schema strictly in your output.
"""



# Convert the Pydantic model's schema to the dictionary format required by the Gemini API
def get_response_schema() -> Dict[str, Any]:
    """Returns the JSON schema dict for structured output required by the Gemini API."""
    schema = ClassificationResult.model_json_schema()
    
    # The Gemini API expects 'type' to be uppercase, and 'propertyOrdering' is useful.
    schema['type'] = schema.pop('type').upper()
    if 'properties' in schema:
        for prop in schema['properties'].values():
            if 'type' in prop:
                prop['type'] = prop['type'].upper()
    schema['propertyOrdering'] = list(schema.get('properties', {}).keys())
    return schema

# --- 3. CORE SERVICE FUNCTIONS ---

def get_gemini_api_payload(objective: str, question: str, existing_categories: List[str]) -> Dict[str, Any]:
    """
    Generates the complete API payload for the Gemini generateContent endpoint.

    Args:
        objective: The user's answer/objective string.
        question: The survey question the user is answering.
        existing_categories: A list of category names that have been used so far.

    Returns:
        A dictionary ready to be converted to JSON for the API request.
    """
    # 1. Validate input using Pydantic (optional, but good practice)
    try:
        ClassificationRequest(answer=objective)
    except ValidationError as e:
        logging.error(f"Input validation error: {e}")
        raise ValueError("Invalid objective format.")

    # Structure the input for the model clearly
    user_prompt = json.dumps({
        "user_answer": objective,
        "question": question,
        "existing_categories": existing_categories
    }, indent=2)

    # 2. Construct the payload
    payload = {
        "contents": [{ 
            "parts": [{ "text": user_prompt }] 
        }],
        "systemInstruction": { 
            "parts": [{ "text": BASE_SYSTEM_PROMPT }] 
        },
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": get_response_schema()
        }
    }
    print(f"\n payload: {payload}")
    return payload

def classify_learning_objective(objective: str, api_url: str, api_key: str, question: str, existing_categories: List[str]) -> ClassificationResult:
    """
    The main function to handle the classification logic (conceptual API call).

    In a real application (e.g., using FastAPI/Flask), this function would
    make the actual HTTP request to the Gemini API and handle parsing.

    Args:
        objective: The user's answer/objective.
        api_url: The Gemini API endpoint URL.
        api_key: Your API key (if required by your environment).
        question: The survey question the user is answering.
        existing_categories: A list of current category names.

    Returns:
        A validated ClassificationResult object.
    """
    try:
        payload = get_gemini_api_payload(objective, question, existing_categories)
        logging.info(f"Sending classification request for: {objective[:30]}...")

        # Make the actual API call to the Gemini API
        full_api_url = f"{api_url}?key={api_key}"
        response = requests.post(full_api_url, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        # Extract the JSON string from the response
        api_response_data = response.json()
        json_string = api_response_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')

        if not json_string or json_string == '{}':
            raise RuntimeError("AI model did not return valid classification content.")

        model_output_data = json.loads(json_string)
        
        # 4. Validate and parse the final result using Pydantic
        classification_result = ClassificationResult(**model_output_data)
        logging.info(f"Classification successful: {classification_result.primaryDomain}")
        
        return classification_result

    except ValidationError as e:
        logging.error(f"Pydantic validation failed for model output: {e}")
        # Handle cases where the model output doesn't match the schema
        raise RuntimeError("AI output failed schema validation.")
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        raise RuntimeError(f"Failed to communicate with the AI service: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during classification: {e}")
        raise

# --- EXAMPLE USAGE (For Testing) ---

if __name__ == "__main__":
    test_objective = "My goal is to understand how neural networks segment tumors in CT scans."
    
    try:
        # Pydantic validates the input data before processing
        # In a real run, these would come from environment variables or a config file
        import os
        GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
        GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent"

        result = classify_learning_objective(
            objective=test_objective, 
            api_url=GEMINI_API_URL, 
            api_key=GEMINI_API_KEY, 
            question="What is your main goal?",
            existing_categories=["Sample Category 1", "Sample Category 2"]
        )
        
        print("\n--- Pydantic Classification Service Output ---")
        print(f"Input Objective: {test_objective}")
        print(f"Domain: {result.primaryDomain}")
        print(f"Confidence: {result.confidenceScore:.2f}")
        print(f"Justification: {result.justification}")
        print("---------------------------------------------")
        
        # Access data as a dictionary (e.g., for JSON response in a web framework)
        print("\nJSON (for API response):")
        print(result.model_dump_json(indent=2))

    except Exception as e:
        print(f"Failed to classify objective: {e}")
