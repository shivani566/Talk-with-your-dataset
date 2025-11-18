import pandas as pd
import numpy as np
import re
from difflib import get_close_matches

DATE_HINTS = ("date", "time", "timestamp", "year", "month", "day")

def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to parse date-like columns in-place (safe).
    """
    for col in df.columns:
        if df[col].dtype == "object" and any(h in col.lower() for h in DATE_HINTS):
            try:
                df[col] = pd.to_datetime(df[col], errors="raise", infer_datetime_format=True)
            except Exception:
                pass
    return df

def guess_date_column(df: pd.DataFrame):
    """
    Return a likely date column name or None.
    """
    # 1) already datetime
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            return c
    # 2) name hints
    for c in df.columns:
        if any(h in c.lower() for h in DATE_HINTS):
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                continue
    return None

def fuzzy_pick_col(user_text: str, df: pd.DataFrame):
    """
    Fuzzy match a column from free text; returns best match or None.
    """
    cols = list(df.columns)
    match = get_close_matches(user_text, cols, n=1, cutoff=0.6)
    return match[0] if match else None

def top_n_categories(df: pd.DataFrame, col: str, n=10):
    return df[col].astype(str).value_counts().head(n).reset_index(names=["value", "count"])

def is_numeric(df: pd.DataFrame, col: str) -> bool:
    return pd.api.types.is_numeric_dtype(df[col])

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()
