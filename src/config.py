import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets ---
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Database settings ---
DB_PATH = "studybuddy_db"
COLLECTION_NAME = "course_notes"


def ensure_db():
    """On a server (a Hugging Face / Streamlit host) the database is not on the disk yet.
    If DB_REPO is set, download it from a private dataset: the first time, and again whenever the dataset
    has a newer version (for example after a new lecture). On my laptop DB_REPO is not set, so nothing happens."""
    repo = os.environ.get("DB_REPO")
    if not repo:
        return
    import shutil
    from huggingface_hub import HfApi, snapshot_download

    token = os.environ.get("DB_READ_TOKEN")
    latest = HfApi().dataset_info(repo, token=token).sha        # id of the newest version of the dataset
    marker = os.path.join(DB_PATH, ".db_revision")
    have = ""
    if os.path.exists(marker) and os.path.exists(os.path.join(DB_PATH, "chroma.sqlite3")):
        with open(marker, encoding="utf-8") as f:
            have = f.read().strip()
    if have != latest:
        shutil.rmtree(DB_PATH, ignore_errors=True)              # start clean, so no old pieces are left behind
        snapshot_download(repo_id=repo, repo_type="dataset", token=token, local_dir=DB_PATH)
        with open(marker, "w", encoding="utf-8") as f:
            f.write(latest)


ensure_db()

# --- Model settings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_PROVIDER = "featherless-ai"
LLM_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# --- File locations ---
LECTURES_DIR = "lectures"
PROCESSED_LOG_PATH = "data/processed_log.json"