"""StudyBuddy AI - chat page, Streamlit version (free hosting on Streamlit Community Cloud).

Run on my laptop:   streamlit run streamlit_app.py
Needs these values (in .env on my laptop, or in the app's Secrets online):
    GROQ_API_KEY      the Groq key
    STUDY_PASSWORD    the password students type
Online only (the notes database is private and is downloaded when the app starts):
    DB_REPO           for example  my-hf-name/studybuddy-db
    DB_READ_TOKEN     a read-only Hugging Face token
"""
import hashlib
import hmac
import json
import os
import re
import threading
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="StudyBuddy AI", page_icon="📚")

# Online, the values live in the app's Secrets. Copy them into the environment so the pipeline finds them.
try:
    for _name, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_name, _value)
except Exception:
    pass                                            # on my laptop there is no secrets file: .env is used

HISTORY_TURNS = 2          # the pipeline sees the last 2 turns, like the terminal chat
MAX_QUESTION_CHARS = 500   # keeps one question from using up the free token limit
CACHE_FILE = "answer_cache.json"
BUSY_MESSAGE = "Busy right now: the free daily limit may be used up. Please try again in a few minutes."
ERROR_START = "Something went wrong on my side"
LIMIT_ERRORS = {"RateLimitError", "InternalServerError", "APIConnectionError", "APITimeoutError"}
EXAMPLES = ["What is temperature in an LLM?", "How does retrieval augmented generation work?"]

st.markdown(
    """
<style>
#hero { text-align: center; padding: 6px 0 4px; }
#hero .title { margin: 0; color: #2B6FA8; font-size: 2.1rem; font-weight: 700; }
#hero .title .hl { background: #FFD54A; color: #1F3556; padding: 0 .35em; border-radius: .4em; }
#hero p { margin: 6px auto 0; max-width: 640px; opacity: .85; }
.chip { display: inline-block; margin: 4px 3px; padding: 3px 11px; border-radius: 999px;
        background: #EDE7FA; font-size: .85rem; }
</style>
<div id="hero">
  <div class="title">Study<span class="hl">Buddy</span> AI</div>
  <p>Ask about the Gen AI and Agentic AI lectures and get an answer in simple words.</p>
  <div>
    <span class="chip">Try: what is self-attention?</span>
    <span class="chip">then: explain it more simply</span>
    <span class="chip">or: explain the code line by line</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ---- Password ----
PASSWORD = os.environ.get("STUDY_PASSWORD")
if not PASSWORD:
    st.error("STUDY_PASSWORD is not set. Add it to .env (laptop) or to the app's Secrets (online).")
    st.stop()

if not st.session_state.get("authed"):
    with st.form("login"):
        typed_password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Open StudyBuddy")
    if submitted:
        if hmac.compare_digest(typed_password.encode(), PASSWORD.encode()):
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


# ---- Load the pipeline once per server (not once per click) ----
@st.cache_resource(show_spinner="Starting up: loading the notes. The first start takes about a minute...")
def load_pipeline():
    from src.rag import pipeline as rag                       # downloads the database first if DB_REPO is set

    size = rag.collection.count()
    # A saved answer belongs to one version of the notes AND one version of the code. If either
    # changes (new lectures, a prompt fix), the cache starts empty, so old wrong answers disappear.
    with open(rag.__file__, "rb") as f:
        code_version = hashlib.md5(f.read()).hexdigest()[:10]
    cache = {"index_size": size, "code_version": code_version, "answers": {}}
    try:
        with open(CACHE_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("index_size") == size and saved.get("code_version") == code_version:
            cache = saved
    except Exception:
        pass
    return {"rag": rag, "cache": cache, "lock": threading.Lock()}


pipe = load_pipeline()
rag, CACHE, LOCK = pipe["rag"], pipe["cache"], pipe["lock"]


def cache_key(question):
    return re.sub(r"\s+", " ", question.lower()).strip(" ?.!")


def save_to_cache(key, answer):
    if not answer.strip() or "covered in your lectures" in answer.lower() or "cut off because" in answer:
        return                                      # never keep refusals, empty or cut-off answers
    CACHE["answers"][key] = answer
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(CACHE, f, ensure_ascii=False)
    except Exception as e:
        print("could not save the cache:", repr(e))


def to_pipeline_history(messages):
    """Chat messages -> (question, answer) pairs. Turns that were refused or failed are dropped."""
    pairs, pending_question = [], None
    for m in messages:
        if m["role"] == "user":
            pending_question = m["content"]
        elif pending_question is not None:
            pairs.append((pending_question, m["content"]))
            pending_question = None
    pairs = [(q, a) for q, a in pairs if a not in (rag.NOT_COVERED_MESSAGE, BUSY_MESSAGE) and not a.startswith(ERROR_START)]
    return pairs[-HISTORY_TURNS:]


def generate_answer(question, history, box):
    """Show the answer in `box` as it is written. Returns (final_text, ok)."""
    key = cache_key(question) if not history else None     # only a first question is safe to reuse
    if key and key in CACHE["answers"]:
        box.markdown(CACHE["answers"][key])                # saved answer: instant and costs nothing
        return CACHE["answers"][key], True
    box.markdown("Waiting for my turn...")                 # one question at a time keeps us under the limit
    with LOCK:
        box.markdown("Searching the notes...")
        try:
            result = rag.answer_question(question, history, stream=True)
            pieces = result.get("answer_stream")
            if pieces is None:                             # for example "not covered": nothing to stream
                box.markdown(result["answer"])
                return result["answer"], True
            raw, last_shown = "", 0.0
            for piece in pieces:
                raw += piece
                now = time.time()
                if now - last_shown > 0.1:
                    box.markdown(rag.clean_answer(raw, fences=True))
                    last_shown = now
            final = rag.clean_answer(raw, fences=True)
            box.markdown(final)                            # final, complete text. Only the answer, never the notes.
            if key:
                save_to_cache(key, final)
            return final, True
        except Exception as e:
            print("answer_question failed:", repr(e))      # shows in the server log
            if type(e).__name__ in LIMIT_ERRORS:
                message = BUSY_MESSAGE                     # the free limit, or Groq is busy
            else:
                message = f"{ERROR_START} ({type(e).__name__}). Please try again, and tell Nitu if it keeps happening."
            box.markdown(message)
            return message, False


# ---- The chat page ----
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.write("One question is answered at a time. If it says it is waiting, please give it a moment.")
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])


def pick_example(text):
    st.session_state.picked = text


picked = st.session_state.pop("picked", None)
typed = st.chat_input("Ask about the lectures...")
question = (typed or picked or "").strip()

if not st.session_state.messages and not question:
    columns = st.columns(len(EXAMPLES))
    for column, example in zip(columns, EXAMPLES):
        column.button(example, on_click=pick_example, args=(example,), use_container_width=True)

if question:
    if len(question) > MAX_QUESTION_CHARS:
        st.warning(f"Please keep your question under {MAX_QUESTION_CHARS} characters.")
    else:
        history = to_pipeline_history(st.session_state.messages)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            box = st.empty()
            final, ok = generate_answer(question, history, box)
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.messages.append({"role": "assistant", "content": final})
