from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocessing import split_features_target
from src.evaluation import evaluate_models, threshold_metrics
from src.risk_scoring import risk_band

st.set_page_config(page_title="Fraud Detection & Risk Analytics", page_icon="💳", layout="wide")
st.title("💳 Intelligent Financial Fraud Detection & Risk Analytics")
st.caption("FinTech machine-learning prototype • Not a production payment-screening system")

with st.sidebar:
    st.header("Analysis settings")
    threshold = st.slider("Fraud review threshold", 0.05, 0.95, 0.50, 0.05)
    high_threshold = st.slider("High-priority threshold", 0.10, 0.99, 0.80, 0.05)
    st.warning("Do not upload real customer identifiers, card numbers, credentials or sensitive financial records.")

st.info("Fraud scores are model estimates for research and prioritization. They are not proof of fraud and should not trigger automatic adverse action.")

uploaded = st.file_uploader("Upload a transaction CSV", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.session_state["fraud_df"] = df
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

df = st.session_state.get("fraud_df")
if df is None:
    st.markdown("### Project workflow")
    st.markdown("Upload transactions → Explore imbalance → Train models → Compare PR-AUC and recall → Review threshold trade-offs → Export analysis.")
else:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions", f"{len(df):,}")
    c2.metric("Features", len(df.columns))
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    st.dataframe(df.head(20), use_container_width=True)

    target = st.selectbox("Select fraud target column", df.columns)
    if st.button("Analyze class distribution"):
        counts = df[target].value_counts(dropna=False).rename_axis("Class").reset_index(name="Transactions")
        st.plotly_chart(px.bar(counts, x="Class", y="Transactions", title="Target class distribution"), use_container_width=True)
        st.dataframe(counts, use_container_width=True)

    if st.button("Train and compare fraud models", type="primary"):
        try:
            X, y = split_features_target(df, target)
            with st.spinner("Training models and evaluating held-out data..."):
                metrics, fitted = evaluate_models(X, y, threshold=threshold)
            st.session_state["fraud_metrics"] = metrics
            st.session_state["fraud_fitted"] = fitted
            st.success("Training complete. The minority class is treated as the positive fraud class.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")

    metrics = st.session_state.get("fraud_metrics")
    fitted = st.session_state.get("fraud_fitted")
    if metrics is not None and fitted:
        st.subheader("Model comparison")
        st.dataframe(metrics.style.format(precision=4), use_container_width=True)
        st.plotly_chart(px.bar(metrics, x="Model", y="PR_AUC", title="Average precision (PR-AUC) comparison"), use_container_width=True)

        chosen = st.selectbox("Inspect model", list(fitted))
        item = fitted[chosen]
        y_true = item["y_test_binary"]
        probs = item["probabilities"]
        tm = threshold_metrics(y_true, probs, threshold)
        a, b, c = st.columns(3)
        a.metric("Precision", f"{tm['precision']:.3f}")
        b.metric("Recall", f"{tm['recall']:.3f}")
        c.metric("F1-score", f"{tm['f1']:.3f}")
        st.write({
            "True positives": tm["true_positives"],
            "False positives": tm["false_positives"],
            "False negatives": tm["false_negatives"],
            "True negatives": tm["true_negatives"]
        })

        scored = item["X_test"].copy()
        scored["fraud_probability"] = probs
        scored["review_band"] = [
            risk_band(p, threshold, high_threshold) for p in probs
        ]
        st.subheader("Held-out transaction review queue (sample)")
        st.dataframe(scored.head(100), use_container_width=True)
        st.download_button("Download scored test transactions",
                           scored.to_csv(index=False),
                           file_name="scored_transactions.csv",
                           mime="text/csv")
        st.download_button("Download model metrics",
                           metrics.to_csv(index=False),
                           file_name="fraud_model_metrics.csv",
                           mime="text/csv")
