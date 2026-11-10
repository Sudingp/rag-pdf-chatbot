# AI evaluation dashboard (Power BI)

One page that shows how the chatbot performs and where it fails, built from the CSVs in
`dashboard/data/` (exported from saved runs with `python -m evals.export_dashboard`).

| File | Grain | Used for |
|---|---|---|
| `rag_cases.csv` | one row per golden-set question (48) | pass rate, categories, failures, latency |
| `metrics.csv` | one row per (evaluation, system, metric) | headline numbers and system comparisons |
| `kg_questions.csv` | one row per list question and system (12) | knowledge graph vs RAG |

## Build it (Power BI Desktop, about 1–2 hours)

1. **Load:** Home > Get data > Text/CSV, load the three files. In Transform data, check the
   types: `passed`, `answerable` and the 0/1 columns as Whole number; `f1`, `latency_ms`,
   `value`, `precision`, `recall` as Decimal number. Close & Apply.
2. **Measures** (Modeling > New measure, on `rag_cases`):
   ```
   Questions        = COUNTROWS(rag_cases)
   Pass rate        = AVERAGE(rag_cases[passed])
   Median latency s = MEDIAN(rag_cases[latency_ms]) / 1000
   Over 10 s        = CALCULATE(COUNTROWS(rag_cases), rag_cases[latency_ms] > 10000)
   ```
   Format `Pass rate` as a percentage.
3. **Top row: cards.** `Questions`, `Pass rate`, `Median latency s`, `Over 10 s`, and a card
   on `metrics[value]` filtered to `metric = hallucination_rate` (format as %).
4. **Pass rate by category:** clustered bar chart, Y = `category`, X = `Pass rate`, sorted
   ascending so the weakest category is on top.
5. **Outcomes:** bar chart, Y = `outcome`, X = count of `case_id`.
6. **Failures table:** table of `case_id`, `category`, `outcome`, `stated_confidence`, with a
   visual filter `passed = 0`. This is the "read the cases, not the totals" view.
7. **Latency:** right-click `latency_ms` > New group > Bin, size 2000; column chart of the bins
   against count of `case_id`.
8. **Knowledge graph vs RAG:** clustered column chart from `kg_questions.csv`, X = `question_id`,
   Y = `recall`, Legend = `system`.
9. **NER vs LLM extraction:** bar chart from `metrics.csv` filtered to evaluation
   `NER: rupee charge amounts` and metric `f1`, Y = `system`, X = `value`.
10. Title the page "Ask the Terms: AI evaluation", save as `dashboard/ask_the_terms.pbix`, and
    export a screenshot (File > Export > PDF, or a screen capture) to `dashboard/dashboard.png`
    for the README.

## Check your numbers

If the dashboard shows these, it is built correctly:

| Visual | Expected |
|---|---|
| Questions / Pass rate | 48 / 93.8% |
| Median latency / Over 10 s | 8.4 s / 15 |
| Hallucination rate | 0% |
| Weakest category | paraphrase, 66.7% (6 cases) |
| Outcomes | CORRECT 35, CORRECT_ABSTAIN 10, PARTIAL 2, WRONG_ABSTAIN 1 |
| Failures table | fact-05, para-03, para-04, all with stated confidence "high" |
| Graph vs RAG recall | graph 100% on all six; RAG 100% only on g5 and g6 |
| NER F1 | pretrained 0%, + EntityRuler 66.7%, + relation rule 100%, LLM 100% |

The failures table carries the main finding: two of the three failures (`para-03`, `para-04`)
are correct answers that the strict scorer marked partial. See `docs/PRODUCT.md`.
