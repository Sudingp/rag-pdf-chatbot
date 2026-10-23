# PDF RAG Chatbot (Offline RAG Project)

This project is a Retrieval-Augmented Generation (RAG) based chatbot that answers questions from uploaded PDF documents using local transformer models and vector search.

## Features
- PDF text extraction
- Text chunking
- SentenceTransformer embeddings
- FAISS vector similarity search
- Transformer QA model
- Interactive Streamlit UI
- Runs fully offline (no API key required)

## Tech Stack
Python, Streamlit, FAISS, SentenceTransformers, Transformers

## How It Works
1. Extract text from PDF
2. Split text into chunks
3. Convert chunks into embeddings
4. Store embeddings in FAISS vector index
5. Retrieve most relevant chunks
6. QA model generates grounded answers

## Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
```
## Answer Models

| Reader | Model | How it answers | Says "not found" when |
| :--- | :--- | :--- | :--- |
| `extractive` (default) | DistilBERT fine-tuned on SQuAD | Copies a span out of the retrieved text | Its confidence is below the threshold |
| `generative` | Flan-T5 base (instruction-tuned LLM, ~250M params) | Writes the answer from a grounded prompt | It replies "unanswerable", or its confidence is below the threshold |
| `azure` | A GPT model on **Azure OpenAI** (e.g. gpt-4o-mini) | Returns **structured JSON** (answer, evidence quote, confidence) and can **call a calculator tool** | It says the answer isn't in the context, a **guardrail** blocks the question or the answer, or Azure's content filter fires |
| `llm` | The same reader on any OpenAI-compatible API: **Gemini**, **Groq** (Llama 3.3), Mistral, OpenRouter, or local **Ollama** | Same as `azure` | Same as `azure` |

The first two run offline with no API key. The extractive reader runs the model directly rather than
through `transformers.pipeline("question-answering")`, because transformers 5 removed that pipeline
(see [INCIDENTS.md](INCIDENTS.md)). Flan-T5 reads at most 512 tokens, so it uses smaller
chunks (500 characters, top 3). Generated answers are checked for **groundedness**: whether the
answer text actually appears in the retrieved context.

## Hosted LLM reader (Azure OpenAI or any OpenAI-compatible provider)

`ragbot/azure_reader.py` adds a hosted LLM with production-style controls:

- **Structured outputs:** the model must reply with JSON matching a strict schema: `answer`, `found_in_context`, `quote` (verbatim evidence) and `confidence`.
- **Function calling:** a `calculate` tool handles arithmetic the document implies but never states (6.80% + 0.50% senior bonus = 7.30%). It evaluates plain arithmetic only and never runs code.
- **Guardrails, in layers:**
  1. **Input:** prompt-injection patterns ("ignore the previous instructions", "you are now...") and empty or over-long questions are refused before any API call.
  2. **Output:** an answer is accepted only if its evidence quote really appears in the retrieved text. Otherwise it becomes "not found", so the model can't invent a source.
  3. **Provider content filter:** a filtered prompt or reply (for example Azure's) becomes a refusal, not a crash.
- **Traceability:** every answer carries a trace (evidence quote, tool calls, guardrail decision), shown in the app under "Why this answer?" and saved in every evaluation report.

- **Provider-agnostic:** the same reader runs on Azure OpenAI (`--reader azure`) or any OpenAI-compatible API (`--reader llm`): Google Gemini, Groq, Mistral, OpenRouter or a local Ollama server. If a provider rejects strict JSON schemas, the reader steps down to JSON mode, then to submitting the answer as a function call (Groq refuses any JSON mode together with tool calling; see INC-002), remembers the mode that worked, and records it in the trace. A reply that is not valid JSON becomes a refusal.

### Set up a free provider (reader `llm`)
1. Get a free API key: **Google AI Studio** (aistudio.google.com, "Get API key") or **Groq** (console.groq.com, "API Keys"). Or install **Ollama** and run `ollama pull qwen2.5:7b` to stay fully offline.
2. Copy `.env.example` to `.env` and uncomment one block (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). `.env` is git-ignored.
3. Pick **Hosted LLM** in the app sidebar, or run `python -m evals.harness --reader llm`.

### Set up Azure (reader `azure`)
1. Create an **Azure OpenAI** resource in the Azure portal, open it in **Azure AI Foundry**, and **deploy** a small chat model (for example `gpt-4o-mini`).
2. Copy `.env.example` to `.env` and fill in the endpoint, key 1, deployment name and API version (from the resource's **Keys and Endpoint** page). `.env` is git-ignored.
3. `pip install -r requirements.txt`, then pick **Azure OpenAI GPT** in the app sidebar, or run
   `python -m evals.harness --reader azure` / `python -m evals.harness --compare-readers` (every reader whose settings exist).

A full harness run is 48 questions: a few cents on a mini model, and within the free tiers of Gemini and Groq.

## Results (live run, 23 October 2026)
Reader `llm` on Groq (`openai/gpt-oss-120b`), chunk 900, overlap 150, top-4, all 48 golden cases:

| Metric | Value |
| :--- | ---: |
| Pass rate | 93.8% |
| Exact match / token F1 (answerable) | 89.5% / 94.5% |
| Retrieval hit rate | 94.6% |
| Hallucination rate (unanswerable questions answered) | 0.0% |
| Guardrail breach rate (4 prompt injections) | 0.0% |
| Reasoning questions (calculator tool) | 2/2 |

Of the 3 failing cases, 2 are correct answers worded differently ("No, there is no fee" vs "No charge";
"Rs 21 per withdrawal" vs "Rs 21 each"), which exact match scores as PARTIAL. The real miss is fact-05,
where the effective date was not in the retrieved chunks.

DeepEval judge (same model, first 10 cases): faithfulness, answer relevancy and contextual recall all
100%. The judge agreed with the exact `grounded` check on 10/10 and with `evidence_retrieved` on 9/10.
The disagreement, dist-03, is a distractor case: the judge credited retrieval because "30 days of
written notice" appears in other chunks, while the sentence that actually answers the question was
not retrieved. The answer was right, possibly by luck, and only the exact check noticed.

The first live run also found a real integration bug (Groq refuses JSON output mode together with
tool calling), now handled and logged as INC-002 in [INCIDENTS.md](INCIDENTS.md).

## Evaluation Harness

The chatbot ships with a harness that measures how often it is right, and how it fails, on a
golden set of 48 questions over a sample banking-terms PDF. The bank in the PDF is fictional.

```
golden.jsonl ──► validate_golden ──► RagPipeline.ask() ──► score (EM, F1, retrieval) ──► classify ──► report
 48 cases        evidence must be        same code the        SQuAD-style metrics        failure mode   markdown + JSON
                 verbatim in the PDF     app uses
```

**Golden set** (`evals/data/golden.jsonl`): factual, numeric, paraphrase, distractor (a similar fact
sits elsewhere in the document), reasoning (needs a calculation), unanswerable questions (the
right behaviour is to say "not found"), and adversarial prompt-injection questions (the right behaviour
is to refuse). There are 48 cases in total. The cases were drafted with an AI assistant and checked
automatically: every evidence sentence must appear verbatim in the PDF, and every expected answer
must appear in its evidence.

**Failure modes** each wrong answer is classified into:

| Mode | Meaning |
| :--- | :--- |
| `RETRIEVAL_MISS` | The sentence with the answer never reached the QA model |
| `EXTRACTION_ERROR` | The right text was retrieved, but the model picked the wrong span |
| `PARTIAL` | The answer overlaps the expected answer but isn't complete |
| `HALLUCINATION` | The question is not answerable from the PDF, but the model answered anyway |
| `WRONG_ABSTAIN` | The answer was in the PDF, but the model said "not found" |
| `REASONING_GAP` | The answer requires a calculation; an extractive model can't produce it |
| `UNGROUNDED` | The answer is wrong and appears nowhere in the retrieved text (made up) |
| `GUARDRAIL_BREACH` | A prompt-injection question was answered instead of refused |

### Run it
```bash
pip install -r requirements.txt
python -m evals.validate_golden                 # check the golden set itself
python -m evals.harness                         # original app settings
python -m evals.harness --sweep                 # compare chunking / top-k / confidence thresholds
python -m evals.harness --save-baseline         # record the current scores
python -m evals.harness --check-baseline        # exit 1 if F1, EM or pass rate drop, or hallucination rises
python -m evals.harness --reader generative     # evaluate Flan-T5 instead of DistilBERT
python -m evals.harness --reader generative --sweep
python -m evals.harness --reader llm            # Gemini / Groq / Ollama (needs .env, see above)
python -m evals.harness --reader azure          # Azure OpenAI reader (needs .env, see above)
python -m evals.harness --compare-readers       # all readers side by side -> evals/results/readers.md
```
Reports are written to `evals/results/`.

### LLM-as-judge with DeepEval
The harness's checks are exact and cheap (does the answer match, is it copied from the context).
`evals/deepeval_eval.py` adds DeepEval's judge-based metrics on top of a finished run, and measures
how often the judge agrees with those exact checks:

| DeepEval metric | What it judges | Compared with |
| :--- | :--- | :--- |
| Faithfulness | every claim in the answer is supported by the retrieved chunks | `grounded` |
| Answer relevancy | the answer addresses the question | - |
| Contextual recall | the retrieved chunks contain what the expected answer needs (judges the retriever) | `evidence_retrieved` |

```bash
python -m evals.harness --reader llm                        # 1. answer the golden set
python -m evals.deepeval_eval evals/results/<run>.json      # 2. judge it -> deepeval_<run>.md
python -m evals.deepeval_eval evals/results/<run>.json --limit 10 --pause 2   # free-tier friendly
```
The judge is any OpenAI-compatible model (`LLM_*` settings, or `JUDGE_*` for a separate judge). It is
reused from the saved run, so the reader is not called again. Rate limits (HTTP 429) are retried with
backoff, a malformed judge reply is recorded as a judge error instead of ending the run, and every
case where the judge and the exact checks disagree is listed for review.

### Tests
```bash
pip install -r requirements-dev.txt
python -m pytest
```
The tests use fake models, so they run in about a second with no downloads. If torch and
transformers are installed, `tests/test_generative_smoke.py` also runs the real Flan-T5 code path
on a tiny randomly initialised T5.

### App settings added
The sidebar now has an **answer model** switch (DistilBERT or Flan-T5), **chunk overlap** and **minimum answer confidence** (below the threshold the app
answers "Not found in the document."). The defaults reproduce the original behaviour.
