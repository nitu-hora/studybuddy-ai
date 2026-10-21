# StudyBuddy AI - Roadmap

A study assistant that answers only from my own lecture notes and says "not covered" otherwise. Principles: free tools only, every decision measured, honest about limits.

## Status

| Phase | What | Status |
|---|---|---|
| 1 | Parse lecture .docx into sections and chunks | Done |
| 2 | Embed and store in ChromaDB, incremental re-index | Done |
| 3 | Retrieval with confidence gate | Done |
| 4 | Answer generation with cleaning | Done |
| 4b | Retrieval fixes (reading lists, multi-query, RRF) | Done |
| 4c | Question rewriting for typos and follow-ups | Done |
| 5 | Evaluation (5-question and 20-question runs) | Done |
| 5b | Speed: simplify route, fast path, batched search | Done (5.6 s to 3.2 s) |
| 6 | Push to GitHub with README | Done |
| 7 | Chat page for a few students (Gradio, then Streamlit) | Done |
| 8 | Free online hosting (Streamlit Community Cloud, private database on Hugging Face) | Done |
| 9 | Answer cache tied to code version, acronym and topic fixes | Done |
| 10 | Pilot with students and collect wrong answers | Next |

## Results so far

On 20 questions, average time per answer fell from 5.6 s to 3.2 s and LLM calls from 3.0 to 2.0. Refusal was 20 of 20 and retrieval hit 10 of 10, unchanged. Details and caveats: [evaluation_notes.md](evaluation_notes.md). Hand-scored correctness is still to be filled in by me.

## Next

- [ ] Hand-score `manual_correct` in the results CSV and put the number in the README
- [x] First clean push: no key, no database, no lecture files
- [x] `streamlit_app.py`: chat page with a password, one question at a time, hosted free online
- [ ] Retest on the hosted app: what is rag, give me an example with code, what is lora
- [ ] Run `python run_check.py low` and read every answer
- [ ] Check LoRA matrix shapes against Lecture 17
- [ ] Test with 5 students, about 10 questions each
- [x] Assignment guides: kept in, solutions included, because they are old assignments only. Never index a current assignment
- [ ] Short feedback form; keep the wrong answers as new test questions

## Future work (cut on purpose, with reasons)

| Idea | Why it is not now |
|---|---|
| Quiz mode (lecture quizzes and AI-made ones) | A separate feature. The retrieval and answer quality come first |
| Telegram bot | Needs a bot host and keys. The web page is enough for 10 students |
| Always-on hosting, Redis cache, semantic cache | Free hosting sleeps after 12 hours idle and its disk is temporary. Fine for a pilot. Needed only for many users |
| Cloud Run or a paid host | Needs a billing card. Consider if the free host is too limited |
| Large pilot (400-500 students) | The free tier allows about 8,000 tokens per minute |
| Re-ranker or relevance cutoff | Would fix invented examples and noisy chunks. Do it if the hand-scored check shows retrieval is the main problem |
| Chunk overlap, hybrid search | Not shown to be the biggest problem yet |
| Skip post-reads and re-index | Small gain for a full re-index |
| Hint-only mode for assignments | Needed only if the guides hold full solutions |
| Per-stage latency in production | Measured offline only for now |

## Known limitations

| Limitation | Evidence | Plan |
|---|---|---|
| Model invents an example when the notes' example is not retrieved | Q7 (Tesla) in the 20-question run | Re-ranker or relevance cutoff |
| The notes can contain errors | A model-size row disagrees with the paper | Fix the notes, then re-index |
| Answers vary between runs | Two different networks for the same question | Accept, or lower the temperature |
| Simplify answers not always much simpler | 4 of 5 at least 20% shorter | Tighter simplify prompt |
| Assignment guides with full solutions are searchable | Assignment details appear in general answers | Intended: old assignments only. Hint-only mode if a current one is ever added |
| Fixed 8 chunks | Noise on narrow questions | Relevance cutoff |
| Look-alike names (RAG vs Ragas) | A short "what is rag" answer described Ragas | Prompt rule and acronym list added; keep testing |
| Free host limits | Sleeps after 12 hours idle; cache resets on restart | Fine for a pilot; Redis or a paid host later |

## Scope

The goal is a project I can explain end to end, with measured results, that a few students can try. It is a pilot-scale tool, not a production system, and I say so. One measured improvement beats three unmeasured features.

## Decisions and alternatives

| Decision | Alternatives | Why this one |
|---|---|---|
| ChromaDB with its default embedding | FAISS, Qdrant, a stronger embedding | Zero setup; measure weaknesses before changing |
| Groq free tier, `gpt-oss-20b` | Hugging Face, a local model | Free and fast; two-line provider switch |
| Remove reading lists at index time | Tag and filter at search time | Simpler; tests showed they aren't needed |
| Multi-query with RRF | Hybrid search, a re-ranker | Cheapest fix for the vocabulary gap |
| Confidence gate before the answer call | Let the model refuse | Free for off-topic questions; fewer made-up answers |
| Fast path and simplify route | Always rewrite and expand | Measured: 5.6 s to 3.2 s with the same quality screens |
| Keep assignment guides, solutions included | Remove them, or filter for students | They are old assignments, and students want the answers |
| Streamlit Community Cloud, private HF dataset for the database | Laptop + share link, Hugging Face Spaces (Gradio now paid), Cloud Run (needs card) | Free, works when my laptop is off, redeploys on git push |
| Answer cache keyed to question + code hash + index size | No cache, or a cache cleared by hand | An old wrong answer once kept coming back after a fix |

## What I learned

- Look at retrieval before blaming the prompt.
- Measure first: my guess that LLM calls dominated the time was wrong, and the per-stage numbers showed it.
- Automatic checks are easy to pass. Hand-scored accuracy is the real number.
- A model repeats what the notes say, including their mistakes.