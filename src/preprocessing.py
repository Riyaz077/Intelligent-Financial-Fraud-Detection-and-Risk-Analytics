"""Preprocessing utilities for transaction-level binary classification."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

def split_features_target(df, target):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")
    clean = df.dropna(subset=[target]).copy()
    X = clean.drop(columns=[target])
    y = clean[target]
    if X.empty:
        raise ValueError("No feature columns remain after selecting the target.")
    if y.nunique() != 2:
        raise ValueError("This starter expects a binary target with exactly two classes.")
    return X, y

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("numeric", num_pipe, numeric))
    if categorical:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("categorical", cat_pipe, categorical))
    if not transformers:
        raise ValueError("No usable feature columns.")
    return ColumnTransformer(transformers=transformers, remainder="drop")
