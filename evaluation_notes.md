# Evaluation notes

Two evaluations, in order:

1. **Part 1, 5-question evaluation** (`eval_five.py`): the first baseline. It found the weak spots and led to versions 2, 3 and 3.1.
2. **Part 2, 20-question benchmark** (`eval_twenty.py`): the main result. It measures speed and quality before and after the fast path (version 4). **Jump to "Version 4: speed (20-question benchmark)" further down for the numbers.**

# Part 1: 5-question evaluation

Baseline run of the pipeline before any of the improvements listed below. The goal was to find out where the system is weak, so I know what to fix first.

**How I ran it:** `eval_five.py` asks 5 test cases and records the rewritten question, the best distance, the retrieved sections, the time and the full answer. Case 5 is a two-turn conversation to test follow-ups.

| # | Question | Retrieved chunk correct? | Answer from notes only? | What I would improve |
|---|---|---|---|---|
| 1 | What is temperature in an LLM? | Yes. All 8 sections are about temperature (Lectures 5, 6, 8) | Mostly. The core explanation matches, but the AI invented example numbers and the T=2 row is wrong (0.55/0.30/0.15 instead of 0.65/0.24/0.11), even though a worked example was retrieved | Prompt: use the notes' own example, never invent numbers |
| 2 | How does retrieval augmented generation work? | Partly. The Lecture 10 chunks were found but ranked 5, 7 and 8, and the top 4 were tangential | Mostly. Step 1 is too advanced for a beginner (IVF, product quantization, routing) and the example was invented | Ranking: add a re-ranker. Prompt: explain the basic flow first |
| 3 | what is atention | Yes. The rewriter fixed the typo, but a "Post-reads" chunk was retrieved | Mostly. The example (cat chased the mouse) was invented while the notes have their own | Data: also skip post-reads and re-index. Gate: distance 0.897 is close to the 1.0 threshold |
| 4 | What is the best way to cook biryani? | n/a. Correctly refused (1.535, weak) | n/a | Speed: the refusal took 5.8 s because the rewrite runs before the gate |
| 5 | What is top-p? / what about top-k? | Turn 1: yes, 6 of 8 sections on top-p. Turn 2: top-k sections ranked first, but 4 of 8 were unrelated (softmax, quizzes, RAG pipeline) because k is fixed at 8 | Turn 1: yes, example arithmetic correct. Turn 2: definition fine, but the example contradicts itself (keeps and drops "glow") and the answer tells the student to go read the lecture, which the prompt forbids | Retrieval: relevance cutoff instead of a fixed 8. Prompt: use the notes' examples. Rewrite: keep "compared to top-p" in the follow-up |

## What worked

- The rewriter fixed the typo ("atention" to "attention") and resolved the follow-up ("what about top-k?" to "What is top-k?").
- The confidence gate refused the off-topic question (1.535) and, in the earlier trace test, the capital of France (1.110).
- The right lectures were found in all four answered questions.

## Pattern I noticed

The right lectures are found almost every time. The weak spots are around the answer: ranking is imperfect, k is always 8 even when few chunks are relevant, and the AI adds its own examples. In 4 of the 5 answers the example looks invented rather than taken from the notes, and two of them contain errors (the temperature numbers and the top-k example).

## Top 3 things to fix

1. **Examples:** use the notes' own examples, never invent numbers, and check the arithmetic.
2. **Retrieval:** add a re-ranker so the best chunk is not 8th, and a relevance cutoff so weak chunks are dropped.
3. **Speed and cost:** search before rewriting, and use fewer tokens per question. The free tier allows 8,000 tokens per minute, and a rate-limit error crashed one of my test runs before I raised `max_retries`.

## Also logged

- Follow-ups like "explain the code" cannot see the previous answer, because history keeps only 300 characters and the answer step gets no history. Fix: a router that passes the last full answer.
- Code questions get AI-written code instead of the course's code. Fix: filter on `has_code`.
- The rewriter outputs a non-breaking hyphen ("top‑p"). Fix: normalize it to "-".
- Data: "Post-reads" sections are not skipped yet. Fix: extend the skip pattern and re-index.
- Quiz sections are indexed. Tag them rather than delete them, since they will help a quiz mode.
- The model sometimes told the student to read a lecture. Decision: answers should be complete enough that this is never needed, so the prompt now forbids mentioning lectures at all.
- Follow-ups like "what about X?" lose the comparison with the previous topic.
- The first full run of the trace test crashed with a 429 rate-limit error. Fixed with `max_retries=6`.

## Changes made after this evaluation (version 2)

- Prompt: reuse the notes' own examples, calculations and code exactly; invent one only if the notes have none, and double-check any numbers.
- Prompt: no mention of lectures, lecture numbers or assignments. The answer should be complete on its own.
- Prompt: bullet points start with "- ", never "*".
- Follow-ups: the previous answer (capped at 1,500 characters) is added to the prompt only when the question refers to it ("the code", "this", "simpler"). The rewriter sees the last 2 turns instead of 4.
- Code questions: the search also fetches up to 4 chunks that contain code (`has_code`) and puts them first.

## What the version 2 chat test showed (9 turns)

**Worked:** the previous answer is used for follow-ups, code chunks are added for code questions, bullets now start with "- ", and the last turn reproduced the notes' `CrossAttnSentiment` code faithfully.

**Problems found:**
- Two answers were cut off (code stopped at `out = self.norm2`, another stopped mid-sentence). Cause: `max_tokens=1500` is too small for a reasoning model on code answers.
- The rewritten question grew longer each turn and picked up old topics.
- A new topic ("embedding model example") was forced into the previous topic.
- "Explain line by line" reprinted the whole code, and differently from the earlier answer.
- A comparison answer claimed the classifier equals the feed-forward layer, which is wrong.
- Invented code passed token ids straight into an encoder layer, which is a bug.
- The phrase "lecture notes" leaked into one answer.
- Italics with `*i*` still appear in some output.

## Changes made after the version 2 chat test (version 3)

- Cut-off detection: if `finish_reason` is "length", a short message is added to the answer. Code answers and follow-ups get `max_tokens=3000`, others 1500.
- Rewriter: use history only when the question cannot be understood without it, leave new topics unchanged, keep the question under 25 words.
- Previous answer block: explain only the lines asked about instead of reprinting all the code.
- Prompt: shorten code when asked for "small" or "simple" code, describe only what is really in the code, never say "the notes" or "the course".

## After the changes (to fill in)

Run on version 3.1: `eval_five.py` plus the 3-turn code test (self-attention, then "write a small code for it", then "explain the code line by line").

| Check | Before | After (v3.1) |
|---|---|---|
| Temperature example numbers correct | No (T=2 wrong) | Yes. I recomputed all 12 probabilities and the 3 entropy values by hand and they match |
| Top-p example arithmetic correct | Yes | Yes (0.19 + 0.17 + 0.15 + 0.14 = 0.65) |
| Top-k example consistent | No (contradicts itself) | No example given, so nothing contradicts. The definition is correct |
| "Explain the code" explains the code just shown | No | Yes. Every line of the class from the previous answer was quoted and explained, to the last line |
| Answer is cut off | Yes (2 of 9 turns) | No (0 of 8 turns) |
| Answer mentions a lecture | Yes | No |
| Answer says "the notes" | Yes | Once: "Key points from the notes" in case 2 (cleaner pattern was missing "from the notes") |
| Raw `*` bullets in output | Yes | No |
| Special hyphens and `\( \)` in the answer | Yes | No |
| Code written by the model runs | Not tested | The small self-attention class is valid PyTorch. An earlier version had a mask with the wrong shape |

**Still open after v3.1**

- "what about top-k?" was rewritten to "What is top-k?", so the search lost the "decoding" context. 4 of 8 retrieved sections were about top-k in RAG or about softmax. The answer was still right, but only because the model guessed the right meaning.
- Case 2 (RAG) pulled in advanced material (re-ranking, Ragas) for a beginner question, and 4 of 8 sections were weak matches.
- Assignment files leak details into general answers ("all 400 review tokens", "GELU").
- The line-by-line explanation said `embed_dim` is "the size of each input vector (and the output of each head)". `embed_dim` is the total size, and each head gets `embed_dim / num_heads`. A small error.
- For "explain the code" the search still runs and finds 8 unrelated sections, although the answer comes from the previous answer. A router that skips the search would save a call.
- Time per question was 12 s for the first question and about 25 s for the rest. Part of that is waiting for the free-tier token limit (8,000 per minute), not compute. Per-stage timing is the next measurement.

# Part 2: 20-question benchmark

## Version 4: speed (20-question benchmark)

**What changed in v4**

- Simplify route: "explain simpler", "give an analogy" and similar short follow-ups rewrite the previous answer directly. No rewrite call and no search, so 1 LLM call.
- Fast path: for the first question of a chat, search the plain question first. If the best distance is below 0.75, skip the rewrite call. Otherwise rewrite and search again. With chat history the rewrite always runs, because follow-ups like "which ones are popular?" have no word that gives them away.
- Batched search: the 3 expansion queries go to ChromaDB in one call instead of three, and the gate search is reused.
- Cleaner: "In the notes, ..." with a capital I is now caught as well.

**Method:** `eval_twenty.py` runs 20 questions I wrote (15 answerable, 5 off-topic; 5 simplify turns, 5 follow-up turns). Same machine, one run each. Compare with `python eval_twenty.py compare baseline fastpath`.

| Check | Baseline (v3.1) | Fast path (v4) |
|---|---|---|
| Average seconds per answered question | 5.6 | 3.2 (43% faster) |
| Average LLM calls per answered question | 3.0 | 2.0 |
| Average rewrite time | 0.7 s | 0.2 s |
| Average search time | 2.7 s (about 1.8 s typical; one 11.6 s outlier) | 0.6 s |
| Rewrite call skipped | n/a | 10 of 15 answered (67%) |
| Refusal correct | 20/20 | 20/20 |
| Retrieval hit (expected lecture in top 8) | 10/10 | 10/10 |
| Keyword screen | 10/10 | 10/10 |
| Follow-up rewrite keeps context | 4/5 | 4/5 |
| Simplify answers at least 20% shorter | 3/5 (weaker check) | 4/5 |
| Answer says "lecture", "the notes" or "the course" | 0/15 | 0/15 |
| Manual: answer correct (my own scoring) | fill in | fill in |

**How to read this**

- Speed improved a lot and the automatic quality screens did not get worse.
- The automatic checks are easy to pass. They show nothing broke. They do not prove the answers are right. The manual column is the real accuracy number.
- The first question took 8.0 s because the embedding model loads on the first call. Without it, the average is about 3.0 s.
- Search was faster by about 3 times against the typical baseline, not 4.5 times, because one slow outlier inflated the baseline average.
- Small test: 20 questions, one laptop, one run each, free-tier waits included.

## Findings from the 20-question run

- **Invented example (Q7, Tesla):** when the notes' own example was not among the 8 retrieved chunks, the model made up an example. The prompt says not to, but it cannot reuse an example it never saw. Fix idea: relevance cutoff or re-ranker so the example chunk is retrieved.
- **The notes can be wrong:** the nomic embedding row in the Lecture 11 notes says "about 500M, GPT-style". The paper (arXiv 2402.01613) says 137M parameters, BERT-based. The model repeated what the notes say, which is correct behaviour for a notes-based assistant. The fix belongs in the notes, not in the code.
- **Answers vary between runs:** the same question asked twice (once with "?" and once without) gave two different small neural networks. Sampling is random, and different chunks can be retrieved for slightly different wording.
- **Simplify answers are not dramatically simpler:** 4 of 5 are shorter by 20% or more, but a reader may still find them close to the original.
- **Small explanation errors remain:** for example `embed_dim` described as the size of each head. The prompt cannot stop this. A manual check can find it.
- **Assignment files leak details** into general answers ("all 400 review tokens", "GELU"). Decision: keep them in the knowledge base for my own use; for the version shared with students, exclude them with a search filter on the lecture name.

## Moved to future work

Quiz mode, a Telegram bot, a large pilot, chunk overlap, a re-ranker, a relevance cutoff, hybrid search, and skipping post-reads with a re-index. None of them was shown to be the biggest problem by the evaluation yet, and each takes time I need for my other projects.