# Suggested notebook workflow

1. `01_Data_Collection_and_Profiling.ipynb` — load dataset, source and schema notes.
2. `02_EDA_and_Class_Imbalance.ipynb` — distributions, missingness and imbalance.
3. `03_Preprocessing_and_Leakage_Checks.ipynb` — feature exclusions and split strategy.
4. `04_Fraud_Model_Training.ipynb` — baseline and ensemble classifiers.
5. `05_Model_Evaluation.ipynb` — PR curves, ROC curves and confusion matrices.
6. `06_Threshold_and_Cost_Analysis.ipynb` — operational threshold trade-offs.
7. `07_Explainability_and_Monitoring.ipynb` — future extension for SHAP, calibration and drift.

Use the reusable modules in `src/` rather than duplicating implementation.
