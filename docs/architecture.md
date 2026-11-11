# System Architecture

1. **Input:** User uploads a permitted transaction CSV.
2. **Profiling:** Inspect schema, missingness and target-class distribution.
3. **Preprocessing:** Numeric values are median-imputed and scaled; categorical fields are mode-imputed and one-hot encoded inside a scikit-learn pipeline.
4. **Training:** Candidate binary classifiers are trained on a stratified training split.
5. **Evaluation:** Compare PR-AUC, ROC-AUC, precision, recall, F1 and confusion counts.
6. **Threshold analysis:** Convert probabilities to review categories using configurable demonstration thresholds.
7. **Dashboard:** Explore model results and export held-out scored records.

No live payment gateway, bank integration, customer identity resolution or automated transaction blocking is implemented.
