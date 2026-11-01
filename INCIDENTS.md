# Incident Log

Failures found in the chatbot, why they happened, and how they were fixed. Newest first.

---

## INC-004: The knowledge-graph scorer marked correct answers wrong

| | |
|---|---|
| **Found** | 2026-11-01, reading the first live `evals.kg_eval` report case by case |
| **Severity** | Medium: the evaluation under-reported both the graph and RAG |
| **Status** | Fixed (`evals/kg_eval.py`, `normalize_answer`); the saved run was re-scored with `--rescore` |
| **Regression test** | `tests/test_kg.py::test_scorer_handles_the_unicode_models_write` |

**What happened.** The report said RAG got question g6 (cheque costs) completely wrong, but its answer was
right: "each additional 25-leaf book Rs 75; stop-payment instruction Rs 100 per cheque". Several graph
answers had the same problem, such as "Rs 500" not being counted.

**Root cause.** The model writes typographic characters: a narrow no-break space in "Rs 500" (U+202F), a
non-breaking hyphen in "stop-payment" (U+2011), the rupee sign instead of "Rs", and curly apostrophes in
"banks' ATMs". The scorer's patterns expected plain ASCII, so correct answers did not match. Two patterns
were also too strict ("first 25-leaf cheque book" did not match "first cheque book").

**Fix.** Answers are normalised before scoring (Unicode NFKC, dashes to "-", curly quotes to straight,
the rupee sign to "Rs"), and the strict patterns were loosened. `--rescore` re-judges a saved run without
calling the model again, so the same answers were scored by the fixed scorer.

**Golden-set review in the same pass.** The model extracted 2 charges the golden set lacked: the first five
free withdrawals at the bank's ATMs and the first three free ones at other banks' ATMs. Both are real, so
they were added and marked `added_after_review`. Before this review the extraction precision was 90.9%.

**Lesson.** Read the cases, not only the totals. A scorer is code and can be wrong too.

---

## INC-003: Knowledge-graph run stopped by Groq's per-minute token limit

| | |
|---|---|
| **Found** | 2026-11-01, first live run of `python -m evals.kg_eval --rag` |
| **Severity** | Medium: the run crashed halfway and nothing was saved |
| **Status** | Fixed (`ragbot/llm_retry.py`) |
| **Regression test** | `tests/test_llm_retry.py` |

**What happened.** Extraction of all 9 chunks finished, then the run crashed at question g3 with
`429 Rate limit reached for model openai/gpt-oss-120b ... tokens per minute (TPM): Limit 8000, Used 5704,
Requested 3521. Please try again in 9.1875s.`

**Root cause.** Each extraction call sends a whole 900-character chunk plus instructions, and the graph
questions add tool results to every follow-up call. Nine back-to-back calls exceeded the free tier's
8,000 tokens per minute. Nothing in the run retried, so one 429 lost all the work done so far.

**Fix.** Every call in the knowledge-graph run (extraction, graph answers and the RAG comparison) now goes
through `RetryingClient`, which waits the time the provider asks for ("try again in 9.1875s", plus 1s)
and falls back to exponential backoff when the error gives no hint. Other errors still fail immediately.

**Detection going forward.** Tests replay Groq's exact error message and check the wait, the backoff and
that non-429 errors are not retried.

---

## INC-001: Fresh install crashes on the first question (transformers 5)

| | |
|---|---|
| **Found** | 2026-10-22, while adding the Flan-T5 reader and testing against the latest `transformers` |
| **Severity** | High: the app loads, then fails on every question |
| **Status** | Fixed (`ragbot/pipeline.py`, `extract_answer`) |
| **Regression tests** | `tests/test_generative_smoke.py::test_extract_answer_*` |

**What happened.** The original `app.py` called
`pipeline("question-answering", model="distilbert-base-cased-distilled-squad")`. `requirements.txt` doesn't
pin `transformers`, so a fresh install now gets 5.x. That raises:

```
KeyError: 'The task question-answering has been removed from transformers.
Use the `text-generation` pipeline with a chat model instead.'
```

**Root cause.** transformers 5.0 removed the extractive `question-answering` pipeline. The app worked on
the 4.x version it was built with and broke on the next install, with no code change on our side.

**Fix.** `extract_answer()` runs the DistilBERT QA model directly. It tokenises question + context into
overlapping 384-token windows (stride 128), masks out question tokens, scores every span up to 30
tokens as p(start) × p(end), and returns the best span with that score. This is the same confidence the
old pipeline reported, so existing `min_score` thresholds still mean the same thing. It works on
transformers 4.x and 5.x.

**Detection going forward.** Smoke tests run the real extraction code on a tiny randomly initialised
DistilBERT (no download) whenever torch/transformers are installed.

**Follow-up.** Re-run `python -m evals.harness` after upgrading, and confirm the scores match the
saved baseline (`--check-baseline`). The span selection is a re-implementation, so small score
differences from the old pipeline are possible.

---

## INC-002: Groq rejects JSON output mode together with tool calling
| | |
|---|---|
| **Found** | 2026-10-23, first live run of `--reader llm` on Groq (openai/gpt-oss-120b) |
| **Severity** | High: every question failed with HTTP 400, so no answers at all |
| **Status** | Fixed |
| **Regression tests** | `test_provider_that_rejects_response_format_with_tools_gets_the_answer_tool`, `test_answer_in_a_call_to_a_made_up_tool_is_recovered`, `test_recovered_answer_still_goes_through_the_quote_guardrail` |

**What happened.** The first question crashed the run with
`400: json mode cannot be combined with tool/function calling`.
**Root cause.** The reader sends a strict JSON schema *and* the calculator tool. Groq refuses
any `response_format` alongside tools, so both the strict request and the existing
`json_object` fallback failed. The fake-client tests only modelled a provider that rejects
strict schemas, not one that rejects the combination.
**Fix.** A third output mode: `response_format` is dropped and the answer itself becomes a
function call (`submit_answer`, with the same schema as its parameters) next to the calculator.
The reader steps down one mode per rejection and remembers the mode that worked, so later
questions cost no failed calls. A first attempt that only moved the schema into the prompt
failed on the next live run: gpt-oss-120b "called" a tool named `json` that did not exist, and
Groq rejected that with `tool_use_failed`. Making the answer a real tool removes the reason to
invent one; as a safety net, an answer found inside a rejected tool call (`failed_generation`)
is recovered and still has to pass the evidence-quote guardrail.
**Detection going forward.** The new regression test replays Groq's exact error message.
The mode in use is recorded in every answer's trace (`structured_output`).

---

## Template

```
## INC-XXX: <one-line summary>
| | |
|---|---|
| **Found** | <date, how> |
| **Severity** | <impact> |
| **Status** | <Open / Fixed in commit abc123> |
| **Regression tests** | <test names> |

**What happened.** <symptom, with the eval run id from evals/results/>
**Root cause.** <why>
**Fix.** <what changed>
**Detection going forward.** <test / harness metric / baseline gate>
```
