# src/router.py
from litellm import Router
from src.config import GEMINI_API_KEY, GROQ_API_KEY
import litellm

# Automatically drop unsupported parameters (like temperature for Gemini)
litellm.drop_params = True

model_list = [
    # Primary Gemini model
    {
        "model_name": "gemini-group",
        "litellm_params": {
            "model": "gemini/gemini-3.8-flash",
            "api_key": GEMINI_API_KEY,
        },
    },
    # Secondary Gemini fallback (often more available during spikes)
    {
        "model_name": "gemini-group",
        "litellm_params": {
            "model": "gemini/gemini-3.6-flash",
            "api_key": GEMINI_API_KEY,
        },
    },
    # Groq fallback (updated to the new free-tier model)
    {
        "model_name": "groq-group",
        "litellm_params": {
            "model": "groq/openai/gpt-oss-120b",  # <-- Updated Groq model
            "api_key": GROQ_API_KEY,
        },
    },
]

router = Router(
    model_list=model_list,
    num_retries=2,
    fallbacks=[{"gemini-group": ["groq-group"]}],
    routing_strategy="simple-shuffle",
)


def get_llm_response(messages):
    response = router.completion(model="gemini-group", messages=messages)
    return response.choices[0].message.content
