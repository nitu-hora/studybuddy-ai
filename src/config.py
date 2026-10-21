import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Database settings ---
DB_PATH = "studybuddy_db"
COLLECTION_NAME = "course_notes"


def ensure_db():
    """On a server (a Hugging Face Space) the database is not on the disk yet.
    If DB_REPO is set, download it once from a private dataset. On my laptop DB_REPO is not set, so nothing happens."""
    repo = os.environ.get("DB_REPO")
    if repo and not os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, repo_type="dataset", token=os.environ.get("DB_READ_TOKEN"), local_dir=DB_PATH)


ensure_db()

# --- Model settings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_PROVIDER = "featherless-ai"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# --- File locations ---
LECTURES_DIR = "lectures"
PROCESSED_LOG_PATH = "data/processed_log.json"