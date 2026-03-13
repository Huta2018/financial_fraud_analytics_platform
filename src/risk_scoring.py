def map_risk_level(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    elif probability < 0.70:
        return "Medium"
    return "High"


def risk_score_percent(probability: float) -> float:
    return round(probability * 100, 2)
