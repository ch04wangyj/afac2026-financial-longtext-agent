"""Financial report numeric normalization and deterministic comparison helpers."""

from __future__ import annotations

import re


_UNIT_MULTIPLIERS = {
    "元": 1,
    "千元": 1_000,
    "万元": 10_000,
    "亿元": 100_000_000,
}



def normalize_numeric_value(text: str, unit: str = "元") -> float:
    cleaned = re.sub(r"[^0-9.\-]", "", text or "")
    if cleaned in {"", ".", "-"}:
        raise ValueError(f"Cannot parse numeric value: {text!r}")
    return float(cleaned) * _UNIT_MULTIPLIERS.get(unit, 1)



def compare_growth(current: float, previous: float, tolerance: float = 1e-9) -> str:
    if current > previous + tolerance:
        return "increase"
    if current < previous - tolerance:
        return "decrease"
    return "flat"



def ratio_exceeds(numerator: float, denominator: float, threshold: float) -> bool:
    if denominator == 0:
        raise ZeroDivisionError("denominator is zero")
    return numerator / denominator > threshold
