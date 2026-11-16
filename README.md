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

## **Installation & Setup**

- Ensure Python 3.10–3.12 is installed.
- Create and activate a Python virtual environment.
- Install the required dependencies using `requirements.txt`.
- Launch the interactive Streamlit dashboard.
- Upload a transaction dataset in CSV format and select the fraud target column.
- Train machine-learning models and compare their evaluation metrics.
- Explore fraud detection results and demonstration risk scores.

## **Command-Line Model Training**

- Execute the training module with the required dataset path.
- Specify the target column (`Class`) for fraud classification.
- Save the trained model artifact in `models/fraud_model.joblib`.
- Use the saved model for subsequent fraud prediction and evaluation.

## Evaluation guidance
Fraud datasets are often highly imbalanced. Accuracy can be misleading: a model that predicts every transaction as legitimate may achieve high accuracy while detecting no fraud. Prioritize PR-AUC / average precision, recall, precision, F1, confusion matrix, and business costs at an explicitly selected threshold. Use temporal splits when transaction timestamps are available; random splits can overstate performance when data is time-dependent.

Do not tune repeatedly against the final test set. Record dataset source/version, target definition, split strategy, random seed, threshold and package versions.

## **Limitations**

- Model predictions are indicative risk scores, not definitive evidence of fraudulent activity.
- False positives may affect legitimate customers, while false negatives may result in financial losses.
- Model performance may vary across different customer and transaction segments.
- Probability calibration and decision thresholds require domain-specific validation.
- The prototype may not meet all requirements for real-world financial deployment.

## **Responsible Use**

- Use this project for educational, research, and fraud analytics purposes.
- Protect sensitive transaction information and never expose payment credentials or personal data.
- Evaluate model fairness and performance across relevant transaction segments.
- Maintain human oversight before taking action based on fraud predictions.
- Ensure secure data handling, continuous monitoring, auditability, and regulatory compliance before production deployment.

## Author

**Mohamed Riyaz S.**  
Integrated M.Tech, Computer Science and Engineering (Business Analytics)  
VIT Chennai
