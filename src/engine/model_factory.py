import os
from typing import Any
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

def get_model() -> Any:
    """
    Senior Pattern: Fallback logic.
    If Gemini (Primary) fails due to rate limits, use Groq (Secondary).
    """
    # Initialize Primary
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_retries=2
    )

    # Initialize Secondary (Llama 3.3 70B via Groq)
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=SecretStr(groq_api_key) if groq_api_key is not None else None
    )

    # Use .with_fallbacks to create a resilient chain
    # If gemini raises an error, it automatically switches to groq
    resilient_model = gemini.with_fallbacks([groq])
    
    return resilient_model