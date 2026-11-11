"""CLI: python -m src.train --csv data/raw/creditcard.csv --target Class"""
import argparse
from pathlib import Path
import joblib
from .preprocessing import load_csv, split_features_target
from .evaluation import evaluate_models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", default="models/fraud_model.joblib")
    args = parser.parse_args()
    df = load_csv(args.csv)
    X, y = split_features_target(df, args.target)
    metrics, fitted = evaluate_models(X, y)
    print(metrics.to_string(index=False))
    best = metrics.iloc[0]["Model"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best]["pipeline"], output)
    print(f"Saved {best} pipeline to {output}")

if __name__ == "__main__":
    main()
