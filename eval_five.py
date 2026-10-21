import sys
import time
sys.stdout.reconfigure(encoding="utf-8")

from src.rag.pipeline import answer_question, HISTORY_TURNS

CASES = [
    ["What is temperature in an LLM?"],
    ["How does retrieval augmented generation work?"],
    ["what is atention"],
    ["What is the best way to cook biryani?"],
    ["What is top-p?", "what about top-k?"],
]

for number, turns in enumerate(CASES, start=1):
    print("=" * 78)
    print(f"CASE {number}")
    history = []
    for question in turns:
        start = time.perf_counter()
        result = answer_question(question, history)
        seconds = time.perf_counter() - start

        print(f"\nYOU ASKED      : {question}")
        print(f"AS REWRITTEN   : {result['standalone_question']}")
        print(f"BEST DISTANCE  : {result['best_distance']:.3f} ({result['confidence']})")
        print(f"TIME           : {seconds:.1f} s")
        if "sources" in result:
            print("RETRIEVED SECTIONS:")
            print(result["sources"])
        print("\nANSWER:")
        print(result["answer"])

        if result["confidence"] == "confident":
            history.append((result["standalone_question"], result["answer"][:300]))
            history = history[-HISTORY_TURNS:]
