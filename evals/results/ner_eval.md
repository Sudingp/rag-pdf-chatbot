# NER for charge amounts

Golden: 11 paid charges in rupees (evals/data/golden_kg.json).

| System | Found | Correct | Precision | Recall | F1 |
|---|---|---|---|---|---|
| spaCy en_core_web_sm (pretrained) | 0 | 0 | 0.0% | 0.0% | 0.0% |
| + currency EntityRuler | 22 | 11 | 50.0% | 100.0% | 66.7% |
| + currency EntityRuler + charge relation rule | 11 | 11 | 100.0% | 100.0% | 100.0% |
| LLM extraction (openai/gpt-oss-120b, forced function calls) | 11 | 11 | 100.0% | 100.0% | 100.0% |

**spaCy en_core_web_sm (pretrained)**
- missed: c01 Rs 150
- missed: c02 Rs 20
- missed: c03 Rs 21
- missed: c05 Rs 75
- missed: c06 Rs 100
- missed: c08 Rs 500
- missed: c09 Rs 200
- missed: c11 Rs 1.5
- missed: c14 Rs 50
- missed: c15 Rs 15
- missed: c17 Rs 250

**+ currency EntityRuler**
- false positive: Rs 5,000: The account must maintain a monthly average balance of Rs 5,000.
- false positive: Rs 100,000: Balances up to Rs 1,00,000 earn 3.25% per annum.
- false positive: Rs 100,000: The portion of a balance above Rs 1,00,000 and up to Rs 10,00,000 earns 3.75% per annum.
- false positive: Rs 1,000,000: The portion of a balance above Rs 1,00,000 and up to Rs 10,00,000 earns 3.75% per annum.
- false positive: Rs 1,000,000: The portion above Rs 10,00,000 earns 4.10% per annum.
- false positive: Rs 40,000: The daily ATM cash withdrawal limit is Rs 40,000.
- false positive: Rs 100,000: The daily limit for UPI transfers is Rs 1,00,000.
- false positive: Rs 10,000: Fixed Deposits Customers with an Everyday Savings Account may open a fixed deposit with a 
- false positive: Rs 500: A recurring deposit allows a fixed monthly instalment of at least Rs 500 for a tenure betw
- false positive: Rs 100: A missed instalment attracts a penalty of Rs 1.50 for every Rs 100 of the instalment.
- false positive: Rs 1,000: SMS alerts are sent for every debit above Rs 1,000; the SMS alert charge is Rs 15 per quar
