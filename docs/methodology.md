# Methodology

## Experimental workflow
- Document the dataset source, license, target definition and time range.
- Remove identifiers that could create leakage or expose sensitive information.
- Inspect class imbalance and missing values.
- Split data before fitting preprocessing steps.
- Prefer temporal train/validation/test splits for time-ordered transactions. If a random split is used for a benchmark, disclose this limitation.
- Train baseline and ensemble classifiers.
- Compare PR-AUC (average precision), recall, precision, F1 and ROC-AUC.
- Evaluate multiple decision thresholds and document false-positive/false-negative trade-offs.
- Select a threshold using explicit operational cost assumptions and validation data, not the final test set.

## Reproducibility
Record dataset version, target mapping, feature exclusions, split dates or random seed, preprocessing, package versions and selected threshold.
