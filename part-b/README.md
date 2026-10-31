# Instamart Customer Copilot - Part B

## Overview

Part B builds customer intelligence from the cleaned Instamart data. It creates interpretable RFM segments, compares clustering diagnostics, trains leakage-aware lapse models, and produces targeted retention recommendations.

The instructor teaching kit is preserved unchanged under `instructor_reference/`. Common generated and processed datasets are included so this branch can be evaluated independently.

## Methodology

1. Calculate recency, frequency, and monetary features from completed orders.
2. Evaluate candidate cluster counts from K=2 through K=8.
3. Use PCA and cluster diagnostics to support the selected segmentation.
4. Compare tuned logistic-regression and random-forest lapse classifiers.
5. Select the model using held-out performance and persist the fitted artifact.
6. Map segment and lapse signals to retention recommendations.

## Results

- Four customer segments were produced.
- Logistic regression was selected as the lapse model.
- Test F1: 0.6878.
- Test ROC-AUC: 0.7830.
- Segment profiles, customer assignments, diagnostics, model results, and retention recommendations are tracked under `outputs/`.

## Run

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.customer_intelligence
pytest -q
```

Open `notebooks/03_customer_intelligence.ipynb` for the executed assignment analysis.

## Conclusion

Part B turns transaction history into explainable customer groups and a leakage-aware lapse-risk signal that can guide differentiated retention actions.
