import pytest
from src.risk_scoring import risk_band

def test_risk_bands():
    assert risk_band(0.2, 0.5, 0.8) == "Lower review priority"
    assert risk_band(0.6, 0.5, 0.8) == "Review"
    assert risk_band(0.9, 0.5, 0.8) == "High review priority"

def test_invalid_probability():
    with pytest.raises(ValueError):
        risk_band(1.2)
