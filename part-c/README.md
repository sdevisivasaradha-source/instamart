# Instamart Customer Copilot - Part C

## Overview

Part C implements privacy-safe support-ticket triage in deterministic mock mode. It redacts personally identifiable information before classification, applies strict input and token limits, validates model-shaped outputs, and routes uncertain or sensitive cases to humans.

The instructor teaching kit is preserved unchanged under `instructor_reference/`. Mock mode requires no credentials or network access.

## Methodology

1. Reject empty and oversized requests.
2. Redact email addresses, phone numbers, and order identifiers.
3. Compare explicit zero-shot and few-shot classification prompts.
4. Enforce retry and token budgets.
5. Validate the returned label schema.
6. Route fraud, Premium, unresolved, and standard cases appropriately.

## Results

- Classified tickets: 147.
- Zero-shot fixture accuracy: 85.71%.
- Few-shot fixture accuracy: 100%.
- Human-review cases: 17.
- Redactions: 55 emails, 91 phone numbers, and 132 order IDs.
- Detailed results and prompt comparisons are tracked under `outputs/`.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.ticket_triage
pytest -q
```

## Conclusion

Part C demonstrates that ticket automation can improve routing while keeping PII outside model prompts and preserving human review for unsafe or uncertain cases.
