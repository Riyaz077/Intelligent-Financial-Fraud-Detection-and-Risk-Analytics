# Limitations and Responsible Use

- Fraud labels are often delayed, incomplete or affected by investigation and chargeback processes.
- Severe class imbalance makes accuracy an inadequate standalone metric.
- A random split can leak temporal patterns and overestimate performance.
- Probability scores may be poorly calibrated and thresholds are not universal.
- False positives can harm legitimate customers; false negatives can cause losses.
- Benchmark data may not represent current fraud patterns, payment channels or customer populations.
- Production systems need secure data governance, encryption, role-based access, audit trails, monitoring, drift detection, human review and incident response.
- Do not use this educational prototype to automatically decline, freeze or block real customer transactions.
