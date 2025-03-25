import os
import logging

# API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")

# Embedding configuration
EMBEDDING_CONFIG = {
    "model": "text-embedding-ada-002",
    "openai_api_key": OPENAI_API_KEY
}

# LLM configuration
LLM_CONFIG = {
    "model_name": "gpt-4o",
    "temperature": 0.3,
    "openai_api_key": OPENAI_API_KEY
}

# Text splitting configuration
CHUNK_CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 100
}

# Paths
VECTOR_STORE_PATH = "./embeddings_store"
DATA_FOLDER = "./data"

# Logging configuration
LOG_FILE = "application.log"
LOG_LEVEL = logging.INFO
