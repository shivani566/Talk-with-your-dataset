import pandas as pd
from .utils import is_numeric

def dataset_overview(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_total": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 3),
    }

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    ms = df.isna().sum().sort_values(ascending=False)
    out = ms[ms > 0].to_frame("missing")
    if out.empty:
        return pd.DataFrame(columns=["column", "missing", "missing_%"])
    out["missing_%"] = (out["missing"] / len(df) * 100).round(2)
    out = out.reset_index().rename(columns={"index": "column"})
    return out

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [c for c in df.columns if is_numeric(df, c)]
    if not num_cols:
        return pd.DataFrame()
    desc = df[num_cols].describe(percentiles=[.05,.25,.5,.75,.95]).T
    desc = desc.rename(columns={"50%": "median"})
    return desc

def categorical_summary(df: pd.DataFrame, top_k=5) -> pd.DataFrame:
    cat_cols = [c for c in df.columns if not is_numeric(df, c)]
    rows = []
    for c in cat_cols:
        vc = df[c].astype(str).value_counts(dropna=False).head(top_k)
        for val, cnt in vc.items():
            rows.append({"column": c, "value": val, "count": int(cnt)})
    return pd.DataFrame(rows)

def summary_markdown(info: dict) -> str:
    lines = [
        f"- **Rows**: {info['rows']}",
        f"- **Columns**: {info['cols']}",
        f"- **Missing (total)**: {info['missing_total']}",
        f"- **Duplicates**: {info['duplicates']}",
        f"- **Memory**: {info['memory_mb']} MB",
    ]
    return "\n".join(lines)
