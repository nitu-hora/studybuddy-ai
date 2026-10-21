import sys, time
sys.stdout.reconfigure(encoding="utf-8")
from src.rag import pipeline as rag

effort = sys.argv[1] if len(sys.argv) > 1 else "low"
rag.REASONING_EFFORT_ANSWER = effort
print("Answer effort:", effort)

REFUSE = "covered in your lectures"
TESTS = [
    ("How much GPU memory does full fine-tuning of a 7B model need, and where does it go?", ["92", "56"], [], False),
    ("Why does LoRA help a company with 200 clients?", ["5.2"], [], False),
    ("What does QLoRA do to the base model?", ["4-bit", "7 GB"], [], False),
    ("What does HHH stand for?", ["Harmless"], ["Humble"], False),
    ("In the \"Where is Kolkata?\" example, what is the probability of the trajectory?", ["0.349"], [], False),
    ("How is DPO different from RLHF?", ["reward model"], [], False),
    ("What dataset is used for multi-hop question answering, and what is in each example?", ["HotpotQA", "distractor"], [], False),
    ("What does BM25 do in the retrieval part?", [], [], False),
    ("What rank r did OpenAI use for GPT-4?", [], [], True),
    ("What is the best GPU to buy for gaming?", [], [], True),
]

passed = 0
for q, must, banned, expect_refuse in TESTS:
    t = time.time()
    res = rag.answer_question(q)
    secs = time.time() - t
    ans = res["answer"]
    low = ans.lower()
    refused = REFUSE in low
    ok = all(w.lower() in low for w in must) and not any(w.lower() in low for w in banned)
    ok = ok and (refused == expect_refuse)
    passed += ok
    print(f"\n{'PASS' if ok else 'FAIL'} | {secs:.1f}s | distance {res['best_distance']:.3f} | words {len(ans.split())} | {q}")
    if not ok or q.startswith("What does BM25"):
        print(ans)
print(f"\nAUTO CHECK: {passed} of {len(TESTS)} passed (this only checks key words; still read the answers)")
