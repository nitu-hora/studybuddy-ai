import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from src.rag import pipeline as rag

q = "How much GPU memory does full fine-tuning of a 7B model need, and where does it go?"
for run in range(1, 4):
    t = time.time()
    res = rag.answer_question(q)
    secs = time.time() - t
    ctx = " ".join(c["text"] for c in res["chunks"])
    print(f"\nRun {run}: {secs:.1f}s | context has '92 GB': {'92 GB' in ctx} | has '8 GB': {'8 GB' in ctx}")
    if run == 1:
        for c in res["chunks"]:
            print("   ", c["lecture"][:35], "|", c["text"][:90].replace("\n", " "))
    print("   Answer starts:", res["answer"].splitlines()[0])
