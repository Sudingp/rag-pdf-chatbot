"""
Builds the evaluation corpus: a FICTIONAL bank's savings-account terms.

"Kestrel Valley Cooperative Bank" does not exist. The document is written to
look like real banking terms (tiers, fees, limits, timelines) so the chatbot
is tested on the kind of dense, number-heavy text a core banking team deals
with. Rebuild the PDF with:

    python evals/data/make_sample_pdf.py
"""

import os

SECTIONS = [
    ("1. About This Document",
     ["This document sets out the terms for the Kestrel Valley Cooperative Bank Everyday Savings Account. "
      "Kestrel Valley Cooperative Bank is a fictional institution created for software testing. "
      "These terms took effect on 1 April 2026 and replace all earlier versions."]),
    ("2. Eligibility and Opening an Account",
     ["Any individual aged 18 years or older who is resident in India may open an Everyday Savings Account. "
      "Minors aged 10 to 17 may hold a junior account operated jointly with a parent or legal guardian.",
      "To open an account, the customer must complete identity verification using a PAN card and one "
      "proof of address. Accounts opened online are activated within 2 working days of successful verification. "
      "A customer may hold at most three Everyday Savings Accounts at any time."]),
    ("3. Minimum Balance",
     ["The account must maintain a monthly average balance of Rs 5,000. If the monthly average balance falls "
      "below this amount, a shortfall charge of Rs 150 plus applicable taxes is applied on the first working "
      "day of the following month.",
      "No shortfall charge applies to junior accounts, or to accounts held by customers aged 60 or above."]),
    ("4. Interest",
     ["Interest is calculated on the end-of-day balance and credited to the account quarterly, in the last "
      "week of June, September, December and March.",
      "Interest rates are tiered by balance. Balances up to Rs 1,00,000 earn 3.25% per annum. The portion of a "
      "balance above Rs 1,00,000 and up to Rs 10,00,000 earns 3.75% per annum. The portion above Rs 10,00,000 "
      "earns 4.10% per annum.",
      "The bank may revise interest rates with 30 days of written notice to the customer."]),
    ("5. Withdrawals and Transfers",
     ["Customers may make up to five free cash withdrawals per calendar month at Kestrel Valley ATMs. Each "
      "additional cash withdrawal costs Rs 20. Withdrawals at other banks' ATMs are free for the first three "
      "per month and cost Rs 21 each after that.",
      "The daily ATM cash withdrawal limit is Rs 40,000. The daily limit for UPI transfers is Rs 1,00,000. "
      "NEFT and RTGS transfers made through internet banking are free of charge."]),
    ("6. Cheque Books",
     ["The first cheque book of 25 leaves in each financial year is free. Each additional cheque book of "
      "25 leaves costs Rs 75. A stop-payment instruction on a cheque costs Rs 100 per cheque."]),
    ("7. Dormant and Inoperative Accounts",
     ["An account with no customer-initiated transaction for 24 consecutive months is classified as "
      "inoperative. Interest continues to be credited to an inoperative account.",
      "To reactivate an inoperative account, the customer must visit any branch with original identity "
      "documents and submit a reactivation form. No charge is levied for reactivation."]),
    ("8. Closing the Account",
     ["An account closed within 14 days of opening is closed free of charge. An account closed after 14 days "
      "and within 12 months of opening attracts a closure charge of Rs 500. No charge applies to accounts "
      "closed after 12 months.",
      "On closure, the remaining balance is transferred to an account nominated by the customer within "
      "3 working days."]),
    ("9. Lost or Stolen Cards",
     ["A lost or stolen debit card must be reported immediately by calling the 24-hour helpline at "
      "1800-555-0199 or by blocking the card in the mobile app. The customer is not liable for unauthorised "
      "transactions reported within 3 working days of the customer receiving the transaction alert.",
      "A replacement debit card costs Rs 200 and is delivered within 7 working days."]),
    ("10. Fixed Deposits",
     ["Customers with an Everyday Savings Account may open a fixed deposit with a minimum amount of Rs 10,000 "
      "for a tenure between 7 days and 10 years. Fixed deposit interest is paid at maturity for tenures up to "
      "180 days, and quarterly for longer tenures.",
      "The fixed deposit rate for a tenure of 1 year to 2 years is 6.80% per annum. Senior citizens aged 60 or "
      "above receive an additional 0.50% per annum on all fixed deposit tenures.",
      "A fixed deposit closed before maturity earns interest at the rate applicable for the period it was held, "
      "minus a premature withdrawal penalty of 1.00%. No interest is paid on fixed deposits closed within "
      "7 days of opening."]),
    ("11. Recurring Deposits",
     ["A recurring deposit allows a fixed monthly instalment of at least Rs 500 for a tenure between 6 months "
      "and 10 years. A missed instalment attracts a penalty of Rs 1.50 for every Rs 100 of the instalment. "
      "If six consecutive instalments are missed, the recurring deposit is closed and the balance is moved to "
      "the savings account."]),
    ("12. Joint Accounts",
     ["An Everyday Savings Account may be held jointly by up to four individuals. The mode of operation may be "
      "either or survivor, former or survivor, or jointly by all holders. A change to the mode of operation "
      "requires written consent from every account holder."]),
    ("13. Nomination",
     ["Every account holder may register one nominee for the account. A nominee who is a minor must have a "
      "guardian named at the time of nomination. A nomination can be added, changed or cancelled at any time "
      "free of charge, and the change takes effect within 5 working days."]),
    ("14. Statements and Alerts",
     ["Account statements are emailed monthly free of charge. A printed statement requested at a branch "
      "costs Rs 50 per statement. SMS alerts are sent for every debit above Rs 1,000; the SMS alert charge is "
      "Rs 15 per quarter. Email alerts are free."]),
    ("15. Standing Instructions",
     ["A standing instruction can be set up to make a recurring payment of a fixed amount on a chosen date. "
      "A standing instruction that fails because of insufficient balance is retried once on the next working "
      "day. Each failed standing instruction attracts a charge of Rs 250. A standing instruction is cancelled "
      "automatically after three consecutive failures."]),
    ("16. Account Freezes and Holds",
     ["The bank may place a hold on an account on the written direction of a court or a government authority. "
      "The customer is informed of the hold within 1 working day, unless the authority instructs otherwise. "
      "During a hold, interest continues to be credited, but withdrawals above the unfrozen amount are refused."]),
    ("17. Changes to These Terms",
     ["The bank will give at least 30 days of notice before any change to these terms that is not in the "
      "customer's favour. Notice is sent by email and published on the bank's website. A customer who does not "
      "accept a change may close the account without a closure charge during the notice period."]),
    ("18. Complaints",
     ["Complaints can be raised at any branch, through the mobile app, or by email to "
      "care@kestrelvalley.example. The bank acknowledges every complaint within 2 working days and aims to "
      "resolve it within 30 days.",
      "If a complaint is not resolved within 30 days, or the customer is not satisfied with the resolution, "
      "the customer may escalate it to the Principal Nodal Officer."]),
]


def build(out_path: str) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    story = [Paragraph("Everyday Savings Account: Terms and Conditions", styles["Title"]), Spacer(1, 12)]
    for heading, paragraphs in SECTIONS:
        story.append(Paragraph(heading, styles["Heading2"]))
        for p in paragraphs:
            story.append(Paragraph(p, styles["BodyText"]))
            story.append(Spacer(1, 6))
    SimpleDocTemplate(out_path, pagesize=A4, title="Everyday Savings Account Terms (fictional)").build(story)
    return out_path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print("Wrote", build(os.path.join(here, "kestrel_savings_terms.pdf")))
