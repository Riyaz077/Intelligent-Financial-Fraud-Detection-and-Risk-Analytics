import pandas as pd
import pytest
from src.preprocessing import split_features_target, build_preprocessor

def test_split():
    df = pd.DataFrame({"amount": [10, 20, 30, 40], "Class": [0, 0, 1, 0]})
    X, y = split_features_target(df, "Class")
    assert list(X.columns) == ["amount"]
    assert y.nunique() == 2

def test_binary_target_required():
    df = pd.DataFrame({"x": [1, 2, 3], "target": [0, 1, 2]})
    with pytest.raises(ValueError):
        split_features_target(df, "target")

def test_preprocessor():
    X = pd.DataFrame({"amount": [10, None, 30], "type": ["A", "B", "A"]})
    transformed = build_preprocessor(X).fit_transform(X)
    assert transformed.shape[0] == 3
