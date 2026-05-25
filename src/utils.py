import math
from typing import Any

import pandas as pd


def to_json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def sanitize_features(features: dict[str, Any]) -> dict[str, Any]:
    return {key: to_json_value(value) for key, value in features.items()}
