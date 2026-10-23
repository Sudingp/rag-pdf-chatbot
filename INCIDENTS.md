# Incident Log

Failures found in the chatbot, why they happened, and how they were fixed. Newest first.

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
