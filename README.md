# Intelligent Financial Fraud Detection and Risk Analytics

An end-to-end FinTech machine-learning starter project for detecting potentially fraudulent transactions, analyzing financial risk patterns, handling severe class imbalance, comparing classification models, and exploring transaction-level risk scores in an interactive Streamlit dashboard.

> **Project status:** This is a development-ready educational/research scaffold, not a production fraud prevention service. It does not include a trained model or measured performance results. Run the experiments on a permitted dataset and report only genuine results.

## Objectives
- Explore transaction data and fraud class imbalance.
- Build leakage-aware preprocessing and classification pipelines.
- Compare baseline and ensemble fraud-detection models.
- Evaluate precision, recall, F1, ROC-AUC and especially PR-AUC.
- Explore decision thresholds and false-positive/false-negative trade-offs.
- Generate transaction risk scores and an interactive analytics dashboard.
- Export evaluation results and scored transactions.

## Key features
- CSV upload and dataset profiling
- Target-column selection and class-imbalance analysis
- Logistic Regression, Random Forest, Extra Trees and HistGradientBoosting
- Optional XGBoost model
- Optional class weighting and training-only resampling with imbalanced-learn
- Stratified train/test split
- Precision, recall, F1, ROC-AUC, average precision (PR-AUC), confusion matrix
- Precision-recall curve and threshold analysis
- Transaction-level fraud probability and configurable review band
- Streamlit dashboard and CSV downloads

## Suggested dataset
- Kaggle Credit Card Fraud Detection: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Alternative datasets: use an authorized, documented transaction dataset with a clearly defined fraud target.

The commonly used European card dataset contains anonymized PCA-derived features plus `Time`, `Amount` and `Class`; `Class=1` indicates fraud. Check dataset terms and licensing before use or redistribution. Do not commit real cardholder information, account numbers, names, addresses or other sensitive data.

## Repository structure
```text
.
├── dashboard/app.py
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── risk_scoring.py
│   └── train.py
├── data/raw/                 # local datasets; ignored by Git
├── data/processed/           # local derived data; ignored by Git
├── models/                   # local model artifacts; ignored by Git
├── reports/figures/
├── docs/
├── tests/
├── requirements.txt
├── config.py
└── README.md
```

## Installation
Python 3.10–3.12 recommended.
```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the dashboard
```bash
streamlit run dashboard/app.py
```
Upload a CSV, select the fraud target column, train models, compare metrics and inspect demonstration risk scores.

## Train from command line
```bash
python -m src.train --csv data/raw/creditcard.csv --target Class --output models/fraud_model.joblib
```

## Evaluation guidance
Fraud datasets are often highly imbalanced. Accuracy can be misleading: a model that predicts every transaction as legitimate may achieve high accuracy while detecting no fraud. Prioritize PR-AUC / average precision, recall, precision, F1, confusion matrix, and business costs at an explicitly selected threshold. Use temporal splits when transaction timestamps are available; random splits can overstate performance when data is time-dependent.

Do not tune repeatedly against the final test set. Record dataset source/version, target definition, split strategy, random seed, threshold and package versions.

## Responsible use and limitations
- A model score is a prioritization signal, not proof of fraud.
- False positives can inconvenience or unfairly exclude legitimate customers; false negatives can cause financial loss.
- Evaluate performance across relevant customer and transaction segments where legally and ethically appropriate.
- Calibrate probabilities and select thresholds with domain stakeholders and documented cost assumptions.
- Production use requires secure data handling, access controls, monitoring, drift detection, auditability, human review, incident response and applicable legal/regulatory review.
- Never publish sensitive transaction data, credentials or payment details.
