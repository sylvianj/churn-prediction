# Customer Churn Predictor

A machine learning web app that predicts the likelihood a telecom customer will churn, built end-to-end from raw data to deployed API.


## Problem

Telecom companies lose millions of dollars to customer churn every year. Acquiring a new customer costs 5–7× more than retaining an existing one. The goal: identify at-risk customers *before* they leave so the business can target retention offers.

## Dataset

[Telco Customer Churn](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) — 7,043 customers with 21 features (contract type, tenure, monthly charges, service usage, etc.).

Target: binary classification — will this customer churn?

Class balance: 26.5% churn, 73.5% no churn (imbalanced).

## Approach

1. **EDA** — Identified contract type, tenure, and monthly charges as the strongest churn signals.
2. **Baseline** — Logistic Regression. Recall on churners: 0.57.
3. **Improvement** — Tried class weighting, Random Forest, and XGBoost. Weighted Logistic Regression won on the metric that matters (recall).
4. **Deployment** — Streamlit app + reusable prediction module in `src/`.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline LR | 0.8038 | 0.6476 | 0.5749 | 0.6091 |
| **Weighted LR** (deployed) | 0.7264 | 0.4909 | **0.7968** | 0.6075 |
| Random Forest | 0.7683 | 0.5545 | 0.6524 | 0.5995 |
| XGBoost | 0.7434 | 0.5120 | 0.7406 | 0.6055 |

**Why recall over accuracy:** Missing a churner costs the company a customer. A false alarm costs a retention offer. The asymmetry means we optimize for recall, accepting lower precision.

The deployed model catches **~80% of churners**.

## Project Structure

```
churn prediction/
├── app/                  # Streamlit web app
├── data/                 # Raw dataset
├── models/               # Trained models & scalers
├── notebooks/            # EDA and modeling notebooks
├── src/                  # Reusable prediction code
├── requirements.txt      # Environment dependencies
└── README.md
```


# Clone
git clone url
cd churn-prediction


python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt


streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

## Key Learnings

- **Simple beats complex on tabular data.** Weighted Logistic Regression matched XGBoost on F1 with a fraction of the complexity.
- **Encoding during inference is a silent bug.** `pd.get_dummies(drop_first=True)` on a single row drops that row's categories, mis-encoding the input. The fix: use `drop_first=False` at inference, then `reindex` to the training columns.
- **Right metric > high metric.** Accuracy said the baseline was best. Recall said the weighted model was best. For churn, recall wins.

## Tech Stack

Python · pandas · scikit-learn · XGBoost · Streamlit · pickle

## License

MIT