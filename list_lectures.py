import sys
sys.stdout.reconfigure(encoding="utf-8")
import collections
import chromadb
from src.config import DB_PATH, COLLECTION_NAME

collection = chromadb.PersistentClient(path=DB_PATH).get_collection(name=COLLECTION_NAME)
metas = collection.get(include=["metadatas"])["metadatas"]
print("Total chunks in the database:", len(metas))
counts = collections.Counter(m.get("lecture", "?") for m in metas)
print("\nChunks per lecture:")
for name, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
    print(f"  {n:5d}  {name}")
