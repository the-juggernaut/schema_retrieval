# Global config and constants
from dotenv import load_dotenv
import os

load_dotenv()

CHUNK_SIZE = "500"
CHUNK_OVERLAP = "100"
MAX_FIELDS_PER_GROUP = "5"
TOP_K_RETRIEVAL = "10"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CONFIDENCE_THRESHOLD = 0.5

OPENAI_MODEL = "gpt-3.5-turbo-instruct"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
