"""
eval_twenty.py - 20-question evaluation for StudyBuddy AI.

Run one evaluation and give it a label:
    python eval_twenty.py baseline

After a change (for example the faster rewriter), run it again with a new label:
    python eval_twenty.py fastpath

Compare two runs:
    python eval_twenty.py compare baseline fastpath

Each run writes eval_results_<label>.csv. Open it in Excel and fill the last two
columns by reading each answer yourself: type yes or no in manual_correct,
and a short reason in manual_note.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import os
import re
import time

# Seconds to wait after each answered question, so the free tier's
# 8,000-tokens-per-minute limit is not hit. Off-topic questions are cheap, so no wait.
SLEEP_AFTER_ANSWER = int(os.environ.get("EVAL_SLEEP", "30"))

# ---------------------------------------------------------------------------
# The 20 questions.
# 5 conversations x 3 turns (factual, follow-up, simplify) = 15 questions,
# plus 5 off-topic questions = 20.
#   lectures    : the lecture numbers where the answer should be found
#   keywords    : at least one should appear in the answer (a rough screen only)
#   rewrite_any : for follow-ups, the rewritten question should contain one of these
# ---------------------------------------------------------------------------
CONVERSATIONS = [
    ("decoding", [
        dict(cat="factual", q="What is top-p sampling?",
             lectures={5, 8}, keywords=["cumulative", "nucleus"]),
        dict(cat="follow-up", q="what about top-k?",
             lectures={5, 8}, keywords=["top-k"],
             rewrite_any=["top-p", "sampling", "decoding", "nucleus"]),
        dict(cat="simplify", q="explain that more simply"),
    ]),
    ("attention", [
        dict(cat="factual", q="What is self-attention?",
             lectures={2, 3, 4}, keywords=["softmax", "weighted"]),
        dict(cat="follow-up", q="and what about cross attention?",
             lectures={2, 3, 4}, keywords=["encoder", "decoder"],
             rewrite_any=["self-attention", "self attention", "transformer"]),
        dict(cat="simplify", q="explain it like I am a beginner"),
    ]),
    ("rag", [
        dict(cat="factual", q="How does retrieval augmented generation work?",
             lectures={10}, keywords=["context", "prompt"]),
        dict(cat="follow-up", q="and what is re-ranking in it?",
             lectures={10, 11, 12}, keywords=["cross-encoder", "rerank", "re-rank"],
             rewrite_any=["rag", "retrieval"]),
        dict(cat="simplify", q="can you make it simpler?"),
    ]),
    ("embeddings", [
        dict(cat="factual", q="What is an embedding model?",
             lectures={11}, keywords=["vector", "dimension"]),
        dict(cat="follow-up", q="which ones are popular?",
             lectures={11}, keywords=["minilm", "bge", "e5", "openai"],
             rewrite_any=["embedding"]),
        dict(cat="simplify", q="explain this in simple words"),
    ]),
    ("temperature", [
        dict(cat="factual", q="What is temperature in an LLM?",
             lectures={5, 6, 8}, keywords=["softmax", "logit", "random", "creativ"]),
        dict(cat="follow-up", q="what happens if I set it to 0?",
             lectures={5, 6, 8}, keywords=["greedy", "deterministic", "same", "argmax"],
             rewrite_any=["temperature"]),
        dict(cat="simplify", q="give me a simpler explanation"),
    ]),
]

OFF_TOPIC = [
    "What is the best way to cook biryani?",
    "What is the capital of France?",
    "Who won the 2018 FIFA World Cup?",
    "How do I fix a leaking tap?",
    "What is the weather in Pune today?",
]

# ---------------------------------------------------------------------------
# Timing. We wrap four functions of rag_from_db.py so that every call adds its
# time to a stage total. Time spent waiting for the free tier (429 retries)
# is inside the LLM calls, so it is counted too.
# ---------------------------------------------------------------------------
STAGE = {"rewrite": 0.0, "expand": 0.0, "search": 0.0, "answer": 0.0}
LLM_CALLS = {"n": 0}


def timed(stage, fn, is_llm=False):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            STAGE[stage] += time.time() - start
            if is_llm:
                LLM_CALLS["n"] += 1
    return wrapper


def norm(text):
    """Lower-case and replace the special hyphen, so checks are fair."""
    return (text or "").replace("‑", "-").lower()


def lecture_numbers(chunks):
    nums = set()
    for c in chunks:
        for n in re.findall(r"Lecture (\d+)", c["lecture"]):
            nums.add(int(n))
    return nums


def yes_no(value):
    if value is None:
        return ""
    return "yes" if value else "no"


COLUMNS = [
    "id", "category", "conversation", "question", "rewritten", "confidence", "best_distance",
    "refusal_ok", "retrieval_hit", "rewrite_ok", "keyword_ok", "no_lecture_mention", "shorter_ok",
    "seconds", "rewrite_s", "expand_s", "search_s", "answer_s", "llm_calls",
    "route", "rewrite_skipped", "answer", "manual_correct", "manual_note",
]


def run_one(rag, qid, cat, conv, question, history, case, previous_answer):
    for k in STAGE:
        STAGE[k] = 0.0
    LLM_CALLS["n"] = 0

    expect_refuse = (cat == "off-topic")
    start = time.time()
    try:
        r = rag.answer_question(question, history)
    except Exception as e:
        print("  ERROR:", e)
        return None, dict(id=qid, category=cat, conversation=conv, question=question,
                          answer="ERROR: " + str(e))
    seconds = time.time() - start

    answer = r["answer"]
    row = dict(
        id=qid, category=cat, conversation=conv, question=question,
        rewritten=r["standalone_question"].replace("‑", "-"),
        confidence=r["confidence"], best_distance=round(r["best_distance"], 3),
        refusal_ok=yes_no(r["confidence"] == ("weak" if expect_refuse else "confident")),
        retrieval_hit="", rewrite_ok="", keyword_ok="", no_lecture_mention="", shorter_ok="",
        seconds=round(seconds, 1),
        rewrite_s=round(STAGE["rewrite"], 1), expand_s=round(STAGE["expand"], 1),
        search_s=round(STAGE["search"], 1), answer_s=round(STAGE["answer"], 1),
        llm_calls=LLM_CALLS["n"], route=r.get("route", "normal"),
        rewrite_skipped=yes_no(r.get("rewrite_skipped", False)),
        answer=answer, manual_correct="", manual_note="",
    )

    if not expect_refuse and r["confidence"] == "confident":
        if case.get("lectures"):
            row["retrieval_hit"] = yes_no(bool(lecture_numbers(r["chunks"]) & case["lectures"]))
        if case.get("keywords"):
            row["keyword_ok"] = yes_no(any(k in norm(answer) for k in case["keywords"]))
        if case.get("rewrite_any"):
            row["rewrite_ok"] = yes_no(any(k in norm(r["standalone_question"]) for k in case["rewrite_any"]))
        if cat == "simplify" and previous_answer:
            # A simpler answer should be clearly shorter: at least 20% shorter.
            row["shorter_ok"] = yes_no(len(answer) < 0.8 * len(previous_answer))
        row["no_lecture_mention"] = yes_no(
            not re.search(r"\b(lecture|the notes|the course)\b", norm(answer)))
    return r, row


def percent(rows, column):
    values = [r[column] for r in rows if r.get(column) in ("yes", "no")]
    if not values:
        return "-"
    return f"{100 * values.count('yes') / len(values):.0f}% ({values.count('yes')}/{len(values)})"


def average(rows, column):
    values = [float(r[column]) for r in rows if r.get(column) not in ("", None)]
    return f"{sum(values) / len(values):.1f}" if values else "-"


def print_summary(rows):
    print("\n" + "=" * 78)
    print(f"{'category':<11}{'n':>3}  {'refusal':<13}{'retrieval':<13}{'rewrite':<13}{'keyword':<13}")
    for cat in ["factual", "follow-up", "simplify", "off-topic"]:
        part = [r for r in rows if r["category"] == cat]
        print(f"{cat:<11}{len(part):>3}  {percent(part, 'refusal_ok'):<13}"
              f"{percent(part, 'retrieval_hit'):<13}{percent(part, 'rewrite_ok'):<13}"
              f"{percent(part, 'keyword_ok'):<13}")
    answered = [r for r in rows if r.get("confidence") == "confident"]
    print(f"\nAnswered questions: {len(answered)}")
    print(f"  average seconds per question : {average(answered, 'seconds')}")
    print(f"  average LLM calls            : {average(answered, 'llm_calls')}")
    print(f"  average seconds by stage     : rewrite {average(answered, 'rewrite_s')}, "
          f"expand {average(answered, 'expand_s')}, search {average(answered, 'search_s')}, "
          f"answer {average(answered, 'answer_s')}")
    print(f"  answers without 'lecture'/'the notes'/'the course': {percent(answered, 'no_lecture_mention')}")
    print(f"  'simpler' answers at least 20% shorter           : {percent(rows, 'shorter_ok')}")
    print(f"  rewrite call skipped (fast path)                 : {percent(rows, 'rewrite_skipped')}")


def run_all(label):
    import rag_from_db as rag

    rag.rewrite_question = timed("rewrite", rag.rewrite_question, True)
    rag.expand_query = timed("expand", rag.expand_query, True)
    rag.generate_answer = timed("answer", rag.generate_answer, True)
    # Time the search. Newer versions search through retrieve_many (retrieve calls it),
    # older versions only have retrieve. We wrap only one of them, so nothing is counted twice.
    if hasattr(rag, "retrieve_many"):
        rag.retrieve_many = timed("search", rag.retrieve_many)
    else:
        rag.retrieve = timed("search", rag.retrieve)

    rows = []
    out_path = f"eval_results_{label}.csv"
    qid = 0
    try:
        for conv, turns in CONVERSATIONS:
            history = []
            previous_answer = None
            for case in turns:
                qid += 1
                print(f"[{qid}/20] ({case['cat']}) {case['q']}")
                r, row = run_one(rag, qid, case["cat"], conv, case["q"], history, case, previous_answer)
                rows.append(row)
                if r is not None and r["confidence"] == "confident":
                    history.append((r["standalone_question"], r["answer"]))
                    history = history[-rag.HISTORY_TURNS:]
                    previous_answer = r["answer"]
                    print(f"    -> {row['seconds']} s, {row['llm_calls']} LLM calls")
                    time.sleep(SLEEP_AFTER_ANSWER)
        for q in OFF_TOPIC:
            qid += 1
            print(f"[{qid}/20] (off-topic) {q}")
            r, row = run_one(rag, qid, "off-topic", "-", q, [], {}, None)
            rows.append(row)
    finally:
        # Save whatever we have, even if the run was stopped half way.
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in COLUMNS})
        print(f"\nSaved {len(rows)} rows to {out_path}")

    print_summary(rows)


def compare(label_a, label_b):
    def load(label):
        with open(f"eval_results_{label}.csv", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    a, b = load(label_a), load(label_b)
    print(f"{'':<34}{label_a:>14}{label_b:>14}")
    for name, fn in [
        ("avg seconds (answered)", lambda rows: average([r for r in rows if r["confidence"] == "confident"], "seconds")),
        ("avg LLM calls (answered)", lambda rows: average([r for r in rows if r["confidence"] == "confident"], "llm_calls")),
        ("avg rewrite seconds", lambda rows: average([r for r in rows if r["confidence"] == "confident"], "rewrite_s")),
        ("simplify answers 20% shorter", lambda rows: percent(rows, "shorter_ok")),
        ("rewrite call skipped", lambda rows: percent([r for r in rows if r["confidence"] == "confident"], "rewrite_skipped")),
        ("refusal correct", lambda rows: percent(rows, "refusal_ok")),
        ("retrieval hit", lambda rows: percent(rows, "retrieval_hit")),
        ("rewrite keeps context", lambda rows: percent(rows, "rewrite_ok")),
        ("keyword screen", lambda rows: percent(rows, "keyword_ok")),
        ("manual: answer correct", lambda rows: percent(rows, "manual_correct")),
    ]:
        print(f"{name:<34}{fn(a):>14}{fn(b):>14}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 2:
        run_all(sys.argv[1])
    else:
        print("Usage: python eval_twenty.py <label>   |   python eval_twenty.py compare <labelA> <labelB>")