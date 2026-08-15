# Pipeline trace — what the four nodes actually do

Every `POST /research` call prints a live trace of the LangGraph state machine to the
server console. These are **real, unedited traces** captured from a local run against
the `sample_docs/` corpus (Groq `llama-3.3-70b-versatile`).

Read them top to bottom: they are the clearest explanation of what the agent is.

---

## Trace 1 — the happy path (exact-match query, graded relevant, single pass)

Query: *"What is the emissions cap and the penalty per excess tonne under Article 24-B?"*

```text
[Planner]  Original: 'What is the emissions cap and the penalty per excess tonne under Article 24-B?'
        -> Rewritten: 'Article 24-B emissions cap regulations and penalty rates per excess tonne of
                       greenhouse gas emissions'

[Retriever] Retrieved 5 documents for the rewritten query
  - article_24_b.txt         (RRF: 0.032266)
  - climate_policy_2024.txt  (RRF: 0.032258)
  - climate_policy_2023.txt  (RRF: 0.031099)
  - article_24_b.txt         (RRF: 0.031025)
  - climate_policy_2024.txt  (RRF: 0.030331)

[Grader] 3/5 docs relevant -> Grade: relevant (retry #0)

[Synthesizer] Answer generated (479 chars)
              Sources: ['article_24_b.txt', 'climate_policy_2024.txt', 'climate_policy_2023.txt']
```

**Why this trace is the whole argument for hybrid retrieval.** `article_24_b.txt` ranks
*first*, ahead of two documents that are far closer in pure embedding space (both climate
policy files talk about emission caps and per-tonne penalties in nearly identical
language). The BM25 arm anchors on the literal token `24-B`; the FAISS arm supplies the
topical context; RRF puts the exact match on top.

The answer that came out of this trace ([screenshot](../screenshots/04-exact-match-article-24b.png))
gives the correct Article 24-B figures — **300 tonnes CO2/yr, $500 per excess tonne** — and
then *explicitly* flags that `climate_policy_2024.txt` states a different cap (400 t / $350)
that does not belong to Article 24-B. A semantic-only retriever routinely collapses those
two into one answer.

---

## Trace 2 — the corrective loop (query with no supporting evidence)

Query: *"What does Article 24-B say about arbitration deadlines?"* — the corpus covers
Article 24-B thoroughly but says nothing about arbitration.

```text
[Planner]  -> 'Article 24-B arbitration deadlines statutory requirements and time limits ...'
[Retriever] Retrieved 5 documents
  - article_24_b.txt (RRF: 0.032266)
  - article_24_a.txt (RRF: 0.031754)
  - article_24_a.txt (RRF: 0.031281)
  - article_24_b.txt (RRF: 0.030769)
  - article_24_b.txt (RRF: 0.030579)
[Grader] 0/5 docs relevant -> Grade: not_relevant (retry #0)

  ... router sends the state back to the Planner ...

[Planner]  -> (same rewrite)
[Retriever] (same 5 documents)
[Grader] 0/5 docs relevant -> Grade: not_relevant (retry #1)
[Router] Max retries reached - forcing synthesis

[Synthesizer] Answer generated (242 chars)
              Sources: ['article_24_b.txt', 'article_24_a.txt']
```

Two things worth noting, and neither is hidden:

1. **Retrieval was still correct.** The right family of documents came back and ranked
   first — the corpus simply does not contain the answer. The grader is measuring
   *answerability*, not retrieval quality.
2. **The loop is bounded and terminates honestly.** After two failed grades the router
   stops retrying and forces synthesis, and the synthesizer's system prompt makes it
   say so rather than invent an answer
   ([screenshot](../screenshots/07-grounded-refusal.png)):
   *"The provided documents do not contain enough information to answer the question..."*

**Known limitation, stated plainly:** the Planner is deterministic (`temperature=0`) and
takes only the *original* query as input, so a retry produces the identical rewrite and
therefore the identical retrieval. The retry loop as currently wired can detect a bad
result but cannot escape one. Making retries productive requires feeding the failure back
into the rewrite — see [Limitations](../README.md#limitations-honest-list).

---

## Trace 3 — partial relevance, majority rule

Query: *"What is the refund policy for defective items?"*

```text
[Retriever]
  - refund_policy.txt        (RRF: 0.032522)
  - refund_policy.txt        (RRF: 0.032522)
  - climate_policy_2024.txt  (RRF: 0.031258)
  - climate_policy_2023.txt  (RRF: 0.030366)
  - climate_policy_2023.txt  (RRF: 0.015625)
[Grader] 2/5 docs relevant -> Grade: not_relevant (retry #1)
[Router] Max retries reached - forcing synthesis
[Synthesizer] Answer generated (613 chars)
              Sources: ['refund_policy.txt', 'climate_policy_2024.txt', 'climate_policy_2023.txt']
```

The two genuinely relevant chunks are both from `refund_policy.txt` and both rank top.
The grader's rule is `relevant_count >= len(documents) / 2`, so 2/5 fails the majority
test even though the retrieved evidence was sufficient. This is the grader being
**conservative**, and it is a deliberate trade-off: a stricter grader costs an extra
retrieval round but never lets a thin evidence set through unflagged.

---

## Reproducing these traces

```bash
python indexer.py          # rebuild FAISS + BM25 from sample_docs/
python main.py             # trace prints to stdout on every /research call
```

Then either use the chat UI at `http://localhost:8000` or:

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the emissions cap and the penalty per excess tonne under Article 24-B?"}'
```
