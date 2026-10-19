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
| 6 | Push to GitHub with README | Next |
| 7 | Chat page for a few students (Gradio) | Next |
| 8 | LinkedIn post | After a few students have used it |

## Results so far

On 20 questions, average time per answer fell from 5.6 s to 3.2 s and LLM calls from 3.0 to 2.0. Refusal was 20 of 20 and retrieval hit 10 of 10, unchanged. Details and caveats: [evaluation_notes.md](evaluation_notes.md). Hand-scored correctness is still to be filled in by me.

## Next

- [ ] Hand-score `manual_correct` in the results CSV and put the number in the README
- [ ] First clean push: no key, no database, no lecture files
- [ ] `app.py`: Gradio chat page with a password, one request at a time, shareable link from my laptop; start with 5 students and about 10 questions each
- [ ] For the student version: exclude assignment files with a search filter if the guides hold full graded solutions
- [ ] Short feedback form and screenshot or recording for the LinkedIn post

## Future work (cut on purpose, with reasons)

| Idea | Why it is not now |
|---|---|
| Quiz mode (lecture quizzes and AI-made ones) | A separate feature. The retrieval and answer quality come first |
| Telegram bot / 24/7 hosting | Needs hosting and keys. A laptop-hosted page is enough for 5 students |
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
| Assignment files leak details | "all 400 review tokens", "GELU" | Filter for the student version |
| Fixed 8 chunks | Noise on narrow questions | Relevance cutoff |

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
| Keep assignment files for now | Remove them | Useful for me; filter them for the student version |
| Laptop-hosted Gradio page | Hosted bot | Free, quick, enough for 5 students |

## What I learned

- Look at retrieval before blaming the prompt.
- Measure first: my guess that LLM calls dominated the time was wrong, and the per-stage numbers showed it.
- Automatic checks are easy to pass. Hand-scored accuracy is the real number.
- A model repeats what the notes say, including their mistakes.