import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# RAG Configuration
DEFAULT_MAX_CHUNK_SIZE = 500
DEFAULT_MIN_CHUNK_SIZE = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.7
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "medical_guidelines"

# Gemini Model Configuration
GEMINI_EMBEDDING_MODEL = "models/embedding-001"
GEMINI_CHAT_MODEL = "gemini-2.0-flash"
EMBEDDING_TASK_TYPE = "retrieval_document"

# PDF Processing Configuration
BATCH_SIZE = 50
DEFAULT_RESULTS_COUNT = 3 