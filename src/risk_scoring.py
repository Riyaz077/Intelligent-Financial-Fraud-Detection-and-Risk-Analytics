"""Transparent, configurable score bands for demonstration only."""
def risk_band(probability, review_threshold=0.50, high_threshold=0.80):
    """Map a model probability to an operational review band, not a fraud verdict."""
    p = float(probability)
    if not 0 <= p <= 1:
        raise ValueError("Probability must be between 0 and 1.")
    if high_threshold <= review_threshold:
        raise ValueError("High threshold must exceed review threshold.")
    if p >= high_threshold:
        return "High review priority"
    if p >= review_threshold:
        return "Review"
    return "Lower review priority"
