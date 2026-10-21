import sys
sys.stdout.reconfigure(encoding="utf-8")
from collections import Counter
from src.rag.pipeline import collection

q = "What is attention in a transformer?"

# 1. what content_type values exist in the database?
data = collection.get(include=["metadatas"])
print(Counter(m["content_type"] for m in data["metadatas"]))
print()

# 2. top 20 results for the question
res = collection.query(query_texts=[q], n_results=20)
print("TOP 20 for:", q)
for i in range(20):
    m = res["metadatas"][0][i]
    print(f"{i+1:>2} {res['distances'][0][i]:.3f} | {m['content_type']:<10} | {m['lecture'][:22]} | {m['section'][:60]}")
print()

# 3. where does the real explanation live? chunks that mention "softmax"
res200 = collection.query(query_texts=[q], n_results=200)
rank = {cid: (r + 1, res200["distances"][0][r]) for r, cid in enumerate(res200["ids"][0])}
hits = collection.get(where_document={"$contains": "softmax"}, include=["metadatas"])
rows = []
for cid, m in zip(hits["ids"], hits["metadatas"]):
    r = rank.get(cid, (999, 9.9))
    rows.append((r[0], r[1], m["lecture"], m["section"]))
rows.sort()
print("Chunks mentioning 'softmax' (the real explanation), by search rank:")
for r in rows[:10]:
    print(f"rank {r[0]:>3} dist {r[1]:.3f} | {r[2][:22]} | {r[3][:60]}")