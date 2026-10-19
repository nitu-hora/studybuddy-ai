import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Database settings ---
DB_PATH = "studybuddy_db"
COLLECTION_NAME = "course_notes"

# --- Model settings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_PROVIDER = "featherless-ai"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# --- File locations ---
LECTURES_DIR = "lectures"
PROCESSED_LOG_PATH = "data/processed_log.json"