import sys
sys.stdout.reconfigure(encoding="utf-8")
import chromadb
from src.config import DB_PATH, COLLECTION_NAME
from src.rag.pipeline import retrieve

col = chromadb.PersistentClient(path=DB_PATH).get_collection(name=COLLECTION_NAME)
data = col.get(include=["documents", "metadatas"])

def find(word):
    hits = [(m["lecture"], d) for d, m in zip(data["documents"], data["metadatas"]) if word.lower() in d.lower()]
    print(f"\n=== Text search for '{word}': {len(hits)} chunks contain it")
    for lec, d in hits[:4]:
        i = d.lower().index(word.lower())
        print("  -", lec[:45], "|", d[max(0, i-80):i+160].replace("\n", " "))

for w in ["Harmless", "clients", "5.2", "GPT-4"]:
    find(w)

for q in ["Why does LoRA help a company with 200 clients?",
          "What does HHH stand for?",
          "What rank r did OpenAI use for GPT-4?"]:
    print("\n=== Search:", q)
    for c in retrieve(q, top_k=5):
        print(f"  {c['distance']:.3f} | {c['lecture'][:40]} | {c['text'][:110]!r}")
