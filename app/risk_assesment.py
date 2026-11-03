"""Project risk scoring; filename kept compatible with the existing project."""
def calculate_risk(impact_score, path_found=False):
    score = max(0, min(100, int(impact_score) + (20 if path_found else 0)))
    if score >= 80:
        level = "Critical"
    elif score >= 60:
        level = "High"
    elif score >= 30:
        level = "Medium"
    else:
        level = "Low"
    return {"score": score, "level": level}
