import chromadb
from sentence_transformers import SentenceTransformer

# Same DB path and collection name your build_index.py already uses
client = chromadb.PersistentClient(path="studybuddy_db")
collection = client.get_collection("course_notes")
model = SentenceTransformer("all-MiniLM-L6-v2")  # adjust if you used a different embedding model

CONFIDENCE_THRESHOLD = 1.0


def classify_confidence(best_distance):
    return "confident" if best_distance < CONFIDENCE_THRESHOLD else "weak"


def test_query(question, n_results=3):
    query_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    distances = results["distances"][0]
    best_distance = distances[0]
    label = classify_confidence(best_distance)

    print(f"Q: {question}  ->  {label}  (best distance={best_distance:.3f})")


if __name__ == "__main__":
    # Queries you EXPECT to be well-covered in your lecture notes
    test_query("what is tokenization")
    test_query("explain cross attention")
    test_query("what is gradient descent")

    # Queries you EXPECT to be weak or not covered
    test_query("what is RLHF")
    test_query("explain quantization")
    test_query("what is chain of thought prompting")