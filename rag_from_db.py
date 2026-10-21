import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass                                        # some servers use a stdout that cannot be reconfigured

import difflib
import os
import re
import time

import chromadb
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, InternalServerError, APIConnectionError

from src.config import DB_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY not found. Add it to the .env file in the project root.")
llm_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key, max_retries=0)

MODEL = "openai/gpt-oss-20b"

# Each model has its own free limit (tokens per minute and per day). If the first model says
# "too many requests", the next one answers instead of the student waiting.
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]


def chat_create(**kwargs):
    """Like llm_client.chat.completions.create, but tries the next model when one is rate limited."""
    last_error = None
    for attempt in range(2):
        for model in MODELS:
            try:
                return llm_client.chat.completions.create(model=model, **kwargs)
            except (RateLimitError, InternalServerError, APIConnectionError) as e:
                last_error = e
                print(f"[limit] {model} failed ({type(e).__name__}), trying the next model")
        if attempt == 0:
            time.sleep(8)   # every model is busy: wait a moment, then try the whole list once more
    raise last_error
TOP_K = 5
CONFIDENCE_THRESHOLD = 1.10

# How long the model may 'think' before it writes. 'low' is faster; 'medium' is the default.
# Rewriting and query-writing are easy jobs, so they always use 'low'.
REASONING_EFFORT_SIDE = "low"
REASONING_EFFORT_ANSWER = "low"

NOT_COVERED_MESSAGE = (
    "This doesn't look like it's covered in your lectures yet.\n"
    "If you meant a different term, check the spelling or try rephrasing, "
    "and I'll search your notes again."
)

FORMAT_RULE = (
    "Formatting rule: write in plain text. Do not use LaTeX and do not use "
    "Markdown emphasis or headings (no **, __, #). Do not use * for bullets or "
    "italics; start bullet points with '- '. If you show code, put it on "
    "its own lines, indented by 4 spaces."
)

# Chunks that contain code are marked has_code=True in the database.
# If check_code_flag.py shows a different value type, change it here.
CODE_FILTER = {"has_code": True}

# Words that suggest the student wants code.
CODE_WORDS = re.compile(
    r"\b(code|implement|implementation|snippet|pytorch|python|script)\b", re.I
)

# Words that suggest the question refers to the previous answer.
FOLLOWUP_WORDS = re.compile(
    r"\b(it|this|that|these|those|above|earlier|previous|again|simpler|simply|more"
    r"|the code|the example|line by line)\b",
    re.I,
)


# Words that mean "explain the previous answer again, but simpler".
SIMPLIFY_WORDS = re.compile(
    r"\b(simpler|simply|simple|beginner|easier|eli5|layman|like i am|like i'm)\b", re.I
)

# Fast path: for the first question of a chat, if the plain question already matches
# the notes this well (distance below this number), we skip the rewrite call.
FAST_PATH_DISTANCE = 0.75

# How many chunks the first search fetches. The gate uses the best one; the merge step reuses all.
SEARCH_K = 8


def retrieve_many(queries, top_k=TOP_K, where=None):
    # One call to ChromaDB for all queries. The texts are embedded together,
    # which is faster than calling ChromaDB once per query.
    results = collection.query(query_texts=queries, n_results=top_k, where=where)

    all_chunks = []
    for q in range(len(queries)):
        chunks = []
        for i in range(len(results["ids"][q])):
            meta = results["metadatas"][q][i]
            chunks.append({
                "id": results["ids"][q][i],
                "text": results["documents"][q][i],
                "lecture": meta["lecture"],
                "section": meta["section"],
                "has_code": meta["has_code"],
                "distance": results["distances"][q][i],
            })
        all_chunks.append(chunks)
    return all_chunks


def retrieve(query, top_k=TOP_K, where=None):
    return retrieve_many([query], top_k, where)[0]


def classify_confidence(best_distance, threshold=CONFIDENCE_THRESHOLD):
    if best_distance < threshold:
        return "confident"
    return "weak"


def format_context(chunks):
    blocks = []
    for n, c in enumerate(chunks, start=1):
        blocks.append(f"[{n}] ({c['lecture']})\n{c['text']}")
    return "\n\n".join(blocks)


def build_system_prompt(context, previous=None):
    if previous:
        previous_block = (
            "\nThe student's previous question and your answer to it. "
            "If the new question refers to it ('it', 'this', 'the code', 'explain more simply'), "
            "it means this earlier answer. When explaining code from it, do not repeat the whole code: "
            "quote only the few lines you are explaining, exactly as written below, and explain each one:\n"
            f"Question: {previous[0]}\n"
            f"Answer:\n{previous[1]}\n"
        )
    else:
        previous_block = ""

    return f"""You are StudyBuddy AI, a teaching assistant for a student's lecture notes.

The numbered notes below come from the student's own course. Write ONE clear, complete answer to the student's question.
- Use only the notes that help answer the question. Ignore notes about other topics.
- Build the answer mainly from the notes. Combine what the different notes say into one explanation: what it is, how it works step by step, and how it is used or shown in an example.
- Use the course's own terms and examples where the notes give them.
- If the notes contain an example, a worked calculation or code for the topic, use it (shorten it if needed) and keep its numbers and code exactly as in the notes. Never invent code or numbers when the notes already have them.
- You may add short clarifications in your own words to make the explanation easier to understand. Do not contradict the notes, and do not add large topics the notes do not cover.
- If the notes only mention the topic in passing, or only use it as an example, say so honestly instead of pretending they explain it.
- Write in simple words for a beginner, in about 3 to 5 short paragraphs. If the notes have no example, add one small example and double-check any numbers you calculate.
- When the student asks for code and a note below already contains code for that topic, copy that code (shortened if needed) and do not write your own version. Write new code only when no note has any, and keep it short and correct: check the shapes of every tensor and every argument you pass.
- If the student asks for small, simple or minimal code, shorten the code from the notes to the essential lines.
- When you describe code or a model, describe only what is actually in it. Do not claim that it contains parts that are not there.
- Give the direct answer in the first sentence, then the explanation.
- Check first: do the notes below really contain what the student asked? If the question names a specific model, number, person, dataset or fact (for example 'what rank did OpenAI use for GPT-4') and that exact thing does not appear in the notes, do not answer from general knowledge and do not borrow a number that belongs to something else. Reply with only these two lines, word for word:
{NOT_COVERED_MESSAGE}
- If the notes spell out what an abbreviation stands for, copy that spelling exactly. Never expand an abbreviation from your own memory.
- Make the answer complete, so the student does not need to look anywhere else. Do not mention lectures, lecture numbers, note numbers, assignments or where to read more, and do not say "the notes" or "the course". Just state the facts.

{FORMAT_RULE}
{previous_block}
Notes:
{context}
"""


def generate_answer(system_prompt, question, max_tokens=1500):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    response = chat_create(
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"reasoning_effort": REASONING_EFFORT_ANSWER},
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    if choice.finish_reason == "length":
        text += "\n\n(This answer was cut off because it got too long. Try asking about one part at a time.)"
    return text


def stream_answer(system_prompt, question, max_tokens=1500):
    """Like generate_answer, but yields the raw text piece by piece while the model writes it."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    stream = chat_create(
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"reasoning_effort": REASONING_EFFORT_ANSWER},
        stream=True,
    )
    finish = None
    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        piece = choice.delta.content
        if piece:
            yield piece
        if choice.finish_reason:
            finish = choice.finish_reason
    if finish == "length":
        yield "\n\n(This answer was cut off because it got too long. Try asking about one part at a time.)"


def is_simplify_request(question, history):
    # Only when there is an earlier answer, the message is short, and it asks for "simpler".
    # Questions that start with "what/define/describe" are normal questions
    # ("what is a simple neural network?"), not requests to simplify.
    return bool(
        history
        and len(question.split()) <= 10
        and SIMPLIFY_WORDS.search(question)
        and not re.match(r"\s*(what|define|describe)\b", question, re.I)
    )


def build_simplify_prompt(previous_question, previous_answer):
    return f"""You are StudyBuddy AI. The student asked for a simpler version of your previous answer.
Rewrite the previous answer for a complete beginner:
- Make it clearly shorter: about 100 words at most.
- Use short sentences and everyday words. Leave out formulas and code unless the student asked for them.
- Use one comparison from daily life to explain the main idea.
- Use only facts that are in the previous answer. Do not add new facts, numbers or claims.
- Do not mention lectures, notes or the course.

{FORMAT_RULE}

Previous question: {previous_question}
Previous answer:
{previous_answer}
"""


def expand_query(question, n=3):
    prompt = (
        "You help search a student's lecture notes on AI and machine learning.\n"
        f"Write {n} short search queries that together find the explanations needed to answer the question below.\n"
        "Each query must name specific sub-concepts and technical terms that a lecture would use. "
        "Do not use generic filler words like 'definition', 'step-by-step' or 'applications'.\n"
        "Cover: what the thing is, its internal parts and mechanics, and how it is used or shown in an example.\n"
        "Fix any spelling mistakes.\n"
        "Example. Question: What is dropout?\n"
        "dropout randomly switches off neurons during training\n"
        "regularization overfitting dropout rate hidden layers\n"
        "dropout turned off at inference time scaling of activations\n\n"
        "Now do the same. Output one query per line, with no numbering and no extra text.\n"
        f"Question: {question}"
    )
    response = chat_create(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
        extra_body={"reasoning_effort": REASONING_EFFORT_SIDE},
    )
    text = response.choices[0].message.content or ""
    queries = []
    for line in text.split("\n"):
        line = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", line).strip()
        if line:
            queries.append(line)
    return queries[:n]


def retrieve_multi(question, first_results, per_query=8, final_k=8):
    try:
        extra = expand_query(question)
    except Exception as e:
        print("  (query writer failed, using the plain question only:", e, ")")
        extra = []

    # first_results is the search for the plain question that we already did for the gate,
    # so we do not search for it a second time.
    result_lists = [first_results]
    if extra:
        result_lists += retrieve_many(extra, top_k=per_query)

    scores = {}
    found = {}
    for chunks in result_lists:
        for rank, c in enumerate(chunks, start=1):
            scores[c["id"]] = scores.get(c["id"], 0) + 1 / (60 + rank)
            found.setdefault(c["id"], c)

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [found[cid] for cid in ranked[:final_k]]


def clean_answer(text):
    # The model sometimes uses special hyphens and spaces that look odd in a terminal.
    text = text.replace("\u2011", "-").replace("\u00a0", " ").replace("\u202f", " ")
    out = []
    in_fence = False
    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            out.append("    " + line)
            continue
        if line.startswith("    "):
            out.append(line)
            continue
        line = re.sub(r"^(\s*)\*\s+", r"\1- ", line)              # "* item" -> "- item"
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])", r"\1", line)  # *italic*
        line = re.sub(r"\\\(|\\\)", "", line)                    # \( and \)
        line = re.sub(r"\\sqrt\{(.*?)\}", lambda m: "\u221a" + m.group(1), line)
        line = re.sub(
            r"\s*,?\s*(as )?(mentioned|described|shown|given|listed) in the (lecture )?notes", "", line)
        line = re.sub(r"\s*\bfrom the (lecture )?notes\b", "", line)
        line = re.sub(r"(^|[.!?]\s+)In the (lecture )?notes,?\s*", r"\1In practice, ", line)
        line = re.sub(r"\bin the (lecture )?notes\b", "here", line, flags=re.I)
        line = re.sub(r"__(.*?)__", r"\1", line)
        line = re.sub(r"^#+\s*", "", line)
        line = re.sub(r"\\\[|\\\]", "", line)
        line = re.sub(r"\\text\{(.*?)\}", r"\1", line)
        line = line.replace("\\times", "*")
        out.append(line)
    return "\n".join(out)


def format_sources(chunks):
    seen = []
    for c in chunks:
        label = f"{c['lecture']} - {c['section']}"
        if label not in seen:
            seen.append(label)
    return "\n".join(f"  - {label}" for label in seen)


DEBUG = True
HISTORY_TURNS = 2


def type_out(text, delay=0.003):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ---- Short requests that have no topic of their own ("write a small code with explanation") ----
# The rewrite step sometimes leaves them unchanged, and then the search finds some unrelated code.
# This check is plain code, not an LLM call, so it never fails: it adds the last real topic.
REQUEST_WORDS = {
    "write", "show", "give", "me", "a", "an", "the", "some", "small", "short", "simple", "tiny", "quick",
    "minimal", "basic", "code", "example", "examples", "explanation", "explain", "with", "and", "also",
    "please", "can", "you", "u", "could", "it", "this", "that", "more", "detail", "details", "in", "python",
    "pytorch", "snippet", "implementation", "implement", "for", "of", "line", "by", "to", "how", "why", "is",
    "calculated", "work", "works", "do", "does", "what", "about", "tell", "using", "step", "full", "complete",
}


def is_topicless_request(question):
    words = re.findall(r"[a-z']+", question.lower())
    if not words or len(words) > 8:
        return False
    for w in words:
        if w in REQUEST_WORDS or difflib.get_close_matches(w, REQUEST_WORDS, n=1, cutoff=0.8):
            continue                                # also accepts small typos like "expalantion"
        return False
    return True


def previous_topic(history):
    """The topic of the most recent earlier question that had a topic of its own."""
    for earlier_question, _ in reversed(history or []):
        if not is_topicless_request(earlier_question):
            topic = re.sub(r"^\s*(what|how|why|who|when|which)\s+(is|are|does|do|was|were)\s+(a|an|the)?\s*",
                           "", earlier_question, flags=re.I)
            return topic.strip(" ?.!")
    return ""


def attach_previous_topic(question, standalone, history):
    if history and is_topicless_request(question):
        topic = previous_topic(history)
        if topic and topic.lower() not in standalone.lower():
            return f"{standalone.rstrip(' ?.!')} about {topic}"
    return standalone


def rewrite_question(question, history):
    if history:
        convo = "\n".join(f"Student: {q}\nAnswer (shortened): {a[:300]}" for q, a in history)
    else:
        convo = "(no earlier conversation)"

    prompt = (
        "Rewrite the student's latest message as ONE standalone question that can be "
        "understood without the conversation.\n"
        "- Fix spelling mistakes.\n"
        "- If the message refers to earlier turns (it, that, this, 'what about...', 'explain more'), "
        "replace those references with the actual topic from the conversation.\n"
        "- If it is already clear and correct, return it unchanged.\n"
        "- Use the conversation only when the message cannot be understood without it. "
        "If the message starts a new topic, keep it as it is and do not add earlier topics to it.\n"
        "- A short request that has no topic of its own, such as 'give me a code example', 'show an example', 'why' or 'how is it calculated', is about the topic of the last turn. Add that topic to it.\n"
        "- Keep the question short, under 25 words.\n"
        "- Do not answer it. Output only the question, on one line.\n\n"
        f"Conversation so far:\n{convo}\n\n"
        f"Latest message: {question}"
    )
    try:
        response = chat_create(
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1000,
            extra_body={"reasoning_effort": REASONING_EFFORT_SIDE},
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        return question

    first_line = text.split("\n")[0].strip().strip('"').strip()
    if not first_line or len(first_line) > 300:
        return question
    return first_line


def answer_question(question, history=None, stream=False):
    history = history or []

    # Route 1: "explain that more simply". Rewrite the previous answer. No search, no rewrite.
    if is_simplify_request(question, history):
        previous_question, previous_answer = history[-1]
        prompt = build_simplify_prompt(previous_question, previous_answer[:4000])
        if stream:
            answer_text, answer_stream = "", stream_answer(prompt, question, 1500)
        else:
            answer_text, answer_stream = clean_answer(generate_answer(prompt, question, 1500)), None
        return {
            "question": question,
            "standalone_question": previous_question,
            "chunks": [],
            "best_distance": 0.0,
            "confidence": "confident",
            "route": "simplify",
            "rewrite_skipped": True,
            "answer": answer_text,
            "answer_stream": answer_stream,
        }

    # Route 2: a normal question.
    # Fast path: for the first question of a chat there is nothing to resolve, only typos.
    # Search with the original question. If it already matches well, skip the rewrite call.
    # With earlier turns we always rewrite, because follow-ups like "which ones are popular?"
    # have no pronoun that we could detect.
    first = None
    if not history:
        first = retrieve(question, top_k=SEARCH_K)
        if first[0]["distance"] >= FAST_PATH_DISTANCE:
            first = None  # weak match: maybe a typo, so let the rewriter try

    if first is not None:
        standalone = question
        rewrite_skipped = True
    else:
        standalone = rewrite_question(question, history)
        standalone = attach_previous_topic(question, standalone, history)
        first = retrieve(standalone, top_k=SEARCH_K)
        rewrite_skipped = False

    best_distance = first[0]["distance"]
    confidence = classify_confidence(best_distance)

    # Safety net for short follow-ups ("give me code example"): if the search came back weak,
    # try once more with the previous topic put in front of the follow-up.
    if confidence == "weak" and history and len(question.split()) <= 8:
        combined = f"{history[-1][0]} {question}"
        retry = retrieve(combined, top_k=SEARCH_K)
        if classify_confidence(retry[0]["distance"]) == "confident":
            standalone, first = combined, retry
            best_distance = first[0]["distance"]
            confidence = "confident"

    result = {
        "question": question,
        "standalone_question": standalone,
        "chunks": first,
        "best_distance": best_distance,
        "confidence": confidence,
        "route": "normal",
        "rewrite_skipped": rewrite_skipped,
    }

    if confidence == "weak":
        result["answer"] = NOT_COVERED_MESSAGE
        return result

    chunks = retrieve_multi(standalone, first)

    # Code questions: also fetch chunks that contain code, and put them first.
    code_mode = False
    if CODE_WORDS.search(standalone):
        code_chunks = retrieve(standalone, top_k=4, where=CODE_FILTER)
        if code_chunks:
            code_mode = True
            seen_ids = {c["id"] for c in code_chunks}
            chunks = code_chunks + [c for c in chunks if c["id"] not in seen_ids][:4]

    result["chunks"] = chunks
    result["code_mode"] = code_mode

    # Follow-ups: give the model the previous answer, so "the code" means that code.
    previous = None
    if history and FOLLOWUP_WORDS.search(question):
        previous = (history[-1][0], history[-1][1][:4000])
    result["used_previous_answer"] = previous is not None

    prompt = build_system_prompt(format_context(chunks), previous)
    # Code answers and follow-ups are longer, so they get a bigger token limit.
    wants_code = bool(CODE_WORDS.search(standalone))
    limit = 3000 if (wants_code or previous) else 1500
    if stream:
        result["answer"] = ""
        result["answer_stream"] = stream_answer(prompt, standalone, limit)
    else:
        raw_answer = generate_answer(prompt, standalone, limit)
        result["answer"] = clean_answer(raw_answer)
    result["sources"] = format_sources(chunks)
    return result


def show_debug(result):
    print(f"\n[debug] you asked           : {result['question']}")
    print(f"[debug] standalone question : {result['standalone_question']}")
    if result.get("route") == "simplify":
        print("[debug] route               : simplify (no search, no rewrite)")
        return
    print(f"[debug] rewrite skipped     : {result.get('rewrite_skipped')}")
    print(f"[debug] best distance       : {result['best_distance']:.3f} ({result['confidence']})")
    if "sources" in result:
        print(f"[debug] code chunks added   : {result['code_mode']}")
        print(f"[debug] previous answer used: {result['used_previous_answer']}")
        print("[debug] retrieved sections:")
        print(result["sources"])


def chat():
    history = []
    print("StudyBuddy AI - ask about your lectures. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            break

        print("Searching your notes...", flush=True)
        try:
            result = answer_question(question, history)
        except Exception as e:
            print("Sorry, something went wrong while answering:", e)
            continue

        if DEBUG:
            show_debug(result)
        print("\nStudyBuddy: ", end="")
        type_out(result["answer"])

        if result["confidence"] == "confident":
            history.append((result["standalone_question"], result["answer"]))
            history = history[-HISTORY_TURNS:]


def run_tests():
    print("--- single questions ---")
    for q in [
        "What is attention in a transformer?",
        "what is atention",
        "What is the best way to cook biryani?",
        "What is the capital of France?",
    ]:
        r = answer_question(q)
        print(f"{q!r} -> {r['standalone_question']!r} | {r['best_distance']:.3f} ({r['confidence']})")

    print("\n--- a conversation (follow-ups) ---")
    history = []
    for q in ["What is temperature in an LLM?", "what about top-p?", "explain that more simply"]:
        r = answer_question(q, history)
        print(f"{q!r} -> {r['standalone_question']!r} | {r['best_distance']:.3f} ({r['confidence']})")
        if r["confidence"] == "confident":
            history.append((r["standalone_question"], r["answer"]))
            history = history[-HISTORY_TURNS:]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        chat()