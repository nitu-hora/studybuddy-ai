import sys
sys.stdout.reconfigure(encoding="utf-8")
from src.rag import pipeline as rag

stats = {"calls": 0, "tokens": 0}
original = rag.llm_client.chat.completions.create

def counting_create(*args, **kwargs):
    response = original(*args, **kwargs)
    stats["calls"] += 1
    if getattr(response, "usage", None):
        stats["tokens"] += response.usage.total_tokens
    return response

rag.llm_client.chat.completions.create = counting_create

for q in ["What is LoRA?", "What does QLoRA do to the base model?", "How is DPO different from RLHF?"]:
    stats["calls"], stats["tokens"] = 0, 0
    rag.answer_question(q)
    print(f"{stats['calls']} calls, {stats['tokens']} tokens | {q}")
