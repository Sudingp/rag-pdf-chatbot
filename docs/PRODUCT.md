# Product brief: Ask the Terms (RAG PDF chatbot)

A chatbot that answers questions about a bank's account terms and charges from the PDF itself,
quoting the sentence it relied on. This brief turns the evaluation results into product
decisions: who it is for, what already works, what to build next and how we will know it
worked. Every number below comes from a saved evaluation run in this repository.

## Users and problems

| User | Situation | What they need |
|---|---|---|
| **Account holder** | Wants to know a fee or rule before acting ("Will I be charged for a stop-payment?") | A short, correct answer with the source, or an honest "not in the document" |
| **Branch / support agent** | Answers the same fee questions all day and must not quote a wrong charge | Fast answers they can verify at a glance, and lists ("which services are free?") |
| **Compliance reviewer** | Signs off what the bot is allowed to say | Evidence that answers come only from the approved document, and a record of failures |

## Where it stands (golden set of 48 hand-checked questions, gpt-oss-120b via Groq, 23 Oct 2026)

| What users feel | Metric | Now |
|---|---|---|
| Answers are right | Pass rate | **93.8%** (45 of 48) |
| It never makes things up | Hallucination rate | **0%** |
| It says "not found" when it should | Unanswerable + prompt-injection cases passed | **10 of 10** |
| It can answer list questions | Recall on 6 list questions | **33.3%** (knowledge-graph tools: 100%) |
| Its confidence means something | AUROC of stated confidence | **0.50**: "high" on all 44 answers, including the wrong ones |
| It is quick | Median latency | **8.4 s**; 15 of 48 answers over 10 s |

**Reading the failures case by case changed the plan.** Only 3 of 48 cases failed:

- `fact-05` "When did these terms take effect?" answered "Not found" (expected 1 April 2026): a
  real miss.
- `para-03` and `para-04` answered "No, there is no fee" and "Rs 21 per withdrawal". Both are
  **correct**; the scorer marked them partial because the wording differs from the expected
  answer. The paraphrase category's 66.7% is mostly a measurement problem, not a model problem.

## What to build next (prioritised)

Ranked by harm to user trust first, then effort. Each story's acceptance criteria are
measurable on the golden set, so "done" is a test result, not an opinion.

| # | Story | Why now | Priority |
|---|---|---|---|
| S1 | List questions across the document | Agents ask these daily; RAG finds a third of the items | High |
| S2 | Confidence that predicts correctness, or no confidence label | A "high" label on wrong answers teaches users to trust mistakes | High |
| S3 | Fix the scorer before the model | 2 of 3 failures are correct answers marked wrong | High |
| S4 | Find document-level facts (effective date) | The only real wrong "not found" | Medium |
| S5 | Answers in under 4 seconds | 8.4 s median feels slow at a counter | Medium |
| S6 | Usage analytics without storing questions | No adoption data yet; questions may contain personal details | Medium |
| S7 | Better exemption extraction | Exemptions F1 57.1% vs charges 100% | Low |

### S1. List questions
*As a support agent, I want to ask "which services are free?" and get every one, so that I don't
give a customer an incomplete answer.*
- Questions asking for a list, a filter or a maximum are routed to the knowledge-graph tools.
- Recall at least 90% on the 6 list questions and on 6 new held-out list questions.
- Single-fact pass rate does not drop below 93.8%.

### S2. Trustworthy confidence
*As an account holder, I want the confidence label to mean something, so that I know when to
double-check.*
- Replace the model's self-reported label with a signal measured on the golden set (for example
  retrieval score), and show it only if its AUROC is at least 0.75.
- Until then, the interface shows the quoted evidence and no confidence label.

### S3. Fix the scorer first
*As the team, we want the pass rate to reflect what users would accept, so that we fix real
problems.*
- Add a meaning-level check (normalised numbers and units, plus an LLM judge) for paraphrase
  cases, with every disagreement with the strict scorer reviewed by hand.
- `para-03` and `para-04` pass; no wrong answer starts passing.

### S4. Document-level facts
*As an account holder, I want to ask when the terms took effect, so that I know they apply to me.*
- `fact-05` passes; the chunk containing the document header is retrievable.
- No other case regresses.

### S5. Speed
*As a support agent, I want an answer in a few seconds, so that I can use it while the customer
waits.*
- Median latency under 4 s and 90th percentile under 8 s on the golden set.
- Pass rate stays at or above 93.8%.

### S6. Usage analytics
*As the product owner, I want to know how the bot is used and where it fails, so that I can
prioritise with data.*
- One structured log event per question: time, answer model, latency, answered / not found /
  refused, guardrail outcome. **Never the question text.**
- A dashboard of daily questions, not-found rate and latency.

### S7. Exemptions
*As a compliance reviewer, I want exemptions (who does not pay a charge) captured as reliably as
charges.*
- Exemption F1 at least 85% against the hand-checked facts (now 57.1%).

## Success measures

- **North star:** share of questions answered correctly with evidence (now 93.8%; 45 of 48).
- **Guardrails that must not move:** hallucination 0%; unanswerable and injection cases 10/10.
- **Adoption (after S6):** questions per day, not-found rate, share of answers over 8 s.

The backlog in `docs/backlog_jira.csv` imports into Jira (Settings > System > External System
Import > CSV). The dashboard data is in `dashboard/data/`; see `dashboard/DASHBOARD.md`.
