import os
import streamlit as st

# Streamlit Cloud uses st.secrets, local uses .env
try:
    # Try loading from Streamlit secrets first
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    COLLECTION_NAME = st.secrets["QDRANT_COLLECTION_NAME"]
except Exception:
    # Fallback for local development
    from dotenv import load_dotenv
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "hr_policy")

# Ensure the key is available for LangChain's default lookups
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
