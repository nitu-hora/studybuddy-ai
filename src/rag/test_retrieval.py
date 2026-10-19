import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import chromadb
from src.config import DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

print("Total chunks in DB:", collection.count())

test_questions = [
    "what is self attention",
    "how does backpropagation work",
    "explain RAG chunking strategy",
    "what is LoRA fine-tuning",
]

for q in test_questions:
    results = collection.query(query_texts=[q], n_results=3)
    print(f"\nQ: {q}")
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  [{meta['lecture']} / {meta['section']}]  (distance={dist:.3f})")
        print("   ", doc[:120].replace(chr(10), " "), "...")

# Diagnostic: does a LoRA-specific chunk even exist in the database?
all_data = collection.get()
lora_sections = [
    (meta["lecture"], meta["section"])
    for meta in all_data["metadatas"]
    if "lora" in meta["section"].lower()
]
print(f"\nSections with 'LoRA' in the title: {len(lora_sections)}")
for lecture, section in lora_sections[:10]:
    print(f"  [{lecture}] {section}")