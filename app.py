"""StudyBuddy AI - chat page for a few students.

Run:  python app.py
It prints a link and asks students for a password.
Needs this in .env:
    STUDY_PASSWORD=choose-a-password
Optional:
    STUDY_USER=studybuddy      (the username, default studybuddy)
    STUDY_SHARE=0              (turn off the Gradio public link, e.g. when using Cloudflare)
"""
import os
import time

import gradio as gr
from dotenv import load_dotenv

# Importing rag_from_db loads the database and the LLM client. It does not start the terminal chat.
import json
import re

from rag_from_db import answer_question, clean_answer, NOT_COVERED_MESSAGE, collection

load_dotenv()

STUDY_USER = os.environ.get("STUDY_USER", "studybuddy")
STUDY_PASSWORD = os.environ.get("STUDY_PASSWORD")
# On a Hugging Face Space (SPACE_ID is set) the Space already has its own public link, so no Gradio share link.
SHARE = os.environ.get("STUDY_SHARE", "1") != "0" and not os.environ.get("SPACE_ID")
if not STUDY_PASSWORD:
    raise RuntimeError("Add STUDY_PASSWORD=your-password to the .env file before starting the page.")

HISTORY_TURNS = 2         # same as the terminal chat: the pipeline sees the last 2 turns
MAX_QUESTION_CHARS = 500  # keeps one question from using up the free token limit


def _text(content):
    """Gradio can send message content as text or as a list of parts. Return plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(_text(part) for part in content)
    if isinstance(content, dict):
        return str(content.get("text", ""))
    return str(content or "")


def to_pipeline_history(chat_history):
    """Turn Gradio's chat history into the (question, answer) pairs that answer_question expects.
    Turns where the bot said 'not covered' are dropped, like in the terminal chat."""
    pairs = []
    pending_question = None
    for item in chat_history or []:
        if isinstance(item, dict):                 # newer Gradio: {"role": ..., "content": ...}
            role, content = item.get("role"), _text(item.get("content"))
            if role == "user":
                pending_question = content
            elif role == "assistant" and pending_question is not None:
                pairs.append((pending_question, content))
                pending_question = None
        else:                                      # older Gradio: [user_text, bot_text]
            user_text, bot_text = _text(item[0]), _text(item[1])
            pairs.append((user_text, bot_text))
    pairs = [(q, a) for q, a in pairs if a != NOT_COVERED_MESSAGE]
    return pairs[-HISTORY_TURNS:]


# ---- Answer cache: the same first question from two students costs the free limit only once ----
CACHE_FILE = "answer_cache.json"


def _cache_key(question):
    return re.sub(r"\s+", " ", question.lower()).strip(" ?.!")


def _load_cache():
    """The cache belongs to one version of the notes. If lectures were added, it starts empty."""
    size = collection.count()
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("index_size") == size:
            return data
    except Exception:
        pass
    return {"index_size": size, "answers": {}}


CACHE = _load_cache()


def _save_to_cache(key, answer):
    if not answer.strip() or "covered in your lectures" in answer.lower() or "cut off because" in answer:
        return                                      # never keep refusals, empty or cut-off answers
    CACHE["answers"][key] = answer
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(CACHE, f, ensure_ascii=False)
    except Exception as e:
        print("could not save the cache:", repr(e))


def respond(message, chat_history):
    question = _text(message).strip()
    if not question:
        yield "Type a question about the lectures."
        return
    if len(question) > MAX_QUESTION_CHARS:
        yield f"Please keep your question under {MAX_QUESTION_CHARS} characters."
        return
    history = to_pipeline_history(chat_history)
    key = _cache_key(question) if not history else None   # only a first question is safe to reuse
    if key and key in CACHE["answers"]:
        yield CACHE["answers"][key]                 # saved answer: instant and costs nothing
        return
    yield "Searching the notes..."                  # shows at once, replaced by the answer below
    try:
        result = answer_question(question, history, stream=True)
        pieces = result.get("answer_stream")
        if pieces is None:                          # for example "not covered": nothing to stream
            yield result["answer"]
            return
        raw, last_shown = "", 0.0
        for piece in pieces:                        # the answer fills in while the model writes it
            raw += piece
            now = time.time()
            if now - last_shown > 0.05:             # about 20 screen updates a second is plenty
                yield clean_answer(raw)
                last_shown = now
        final = clean_answer(raw)
        yield final                                 # final, complete text. Only the answer, never the notes.
        if key:
            _save_to_cache(key, final)
    except Exception as e:
        print("answer_question failed:", repr(e))   # shows in my terminal, not to the student
        yield "Busy right now: the free daily limit may be used up. Please try again in a few minutes."


# ---- Look and feel: sky blue, soft yellow, light lavender ----
CSS = """
:root {
  --sky: #7CC6F2;  --sky-light: #E3F4FD;  --sky-deep: #2B6FA8;
  --lavender: #EDE7FA;  --lavender-deep: #B9A8E8;
  --sun: #FFD54A;  --sun-deep: #F2B705;
  --ink: #1F3556;
}
body, gradio-app, .gradio-container {
  background: linear-gradient(160deg, var(--sky-light) 0%, #F4F0FD 55%, #FFF8DC 100%) !important;
  color: var(--ink) !important;
}
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }
.gradio-container *:not(button):not(svg):not(path) { color: var(--ink); }
footer { display: none !important; }
#hero { text-align: center; padding: 22px 12px 6px; }
#hero h1 { font-size: 2.1rem; margin: 0; color: var(--sky-deep) !important; letter-spacing: .3px; }
#hero h1 span { background: var(--sun); color: var(--ink) !important; padding: 0 .35em; border-radius: .4em; }
#hero p { margin: 8px auto 0; max-width: 640px; opacity: .85; }
.chip { display:inline-block; margin:4px 3px; padding:3px 11px; border-radius:999px;
        background: var(--lavender); font-size: .85rem; }
/* chat bubbles: remove Gradio's outer box, style only the inner bubble */
[data-testid="user"], [data-testid="bot"] { background: transparent !important; border: none !important; }
.user.message, .bot.message { background: transparent !important; border: none !important; box-shadow: none !important; }
.user.message > .panel-full-width { background: var(--lavender) !important; padding: 10px 16px !important;
        border: 1px solid var(--lavender-deep) !important; border-radius: 16px !important; }
.bot.message > .panel-full-width  { background: #FFFFFF !important; padding: 10px 16px !important;
        border: 1px solid var(--sky) !important; border-radius: 16px !important; }
.gradio-container .block { background: rgba(255,255,255,.55) !important; border-color: var(--sky-light) !important; }
/* send button and input */
.submit-button, button.primary { background: var(--sun) !important; border: 1px solid var(--sun-deep) !important; }
.submit-button svg, button.primary svg { color: var(--ink) !important; fill: var(--ink) !important; }
.submit-button:hover, button.primary:hover { background: var(--sun-deep) !important; }
textarea, input[type="text"] { background: #FFFFFF !important; border: 1px solid var(--sky) !important; }
textarea:focus { border-color: var(--sky-deep) !important; box-shadow: 0 0 0 2px var(--sky-light) !important; }
button.example, .example { background: var(--lavender) !important; border: 1px solid var(--lavender-deep) !important; }
.examples-holder, .examples { background: transparent !important; }
"""

HERO = """
<div id="hero">
  <h1>Study<span>Buddy</span> AI</h1>
  <p>Ask about the Gen AI and Agentic AI lectures and get an answer in simple words.</p>
  <div>
    <span class="chip">Try: what is self-attention?</span>
    <span class="chip">then: explain it more simply</span>
    <span class="chip">or: explain the code line by line</span>
  </div>
  <p style="font-size:.85rem;opacity:.7">One question is answered at a time. If it says it is waiting, please give it a moment.</p>
</div>
"""

# Always show the light sky-blue look, even when the student's browser is in dark mode.
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); }"

THEME = gr.themes.Soft(primary_hue="sky", secondary_hue="amber", neutral_hue="slate")

with gr.Blocks(title="StudyBuddy AI") as demo:
    gr.HTML(HERO)
    gr.ChatInterface(
        fn=respond,
        examples=["What is temperature in an LLM?", "How does retrieval augmented generation work?"],
    )

if __name__ == "__main__":
    # One request at a time keeps us under the free-tier token limit.
    demo.queue(default_concurrency_limit=1, max_size=10)
    demo.launch(share=SHARE, auth=(STUDY_USER, STUDY_PASSWORD), theme=THEME, css=CSS, js=FORCE_LIGHT)