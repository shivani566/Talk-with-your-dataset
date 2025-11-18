import pandas as pd
import plotly.express as px
from .utils import is_numeric

def build_chart(df: pd.DataFrame, chart_type: str, x: str = None, y: str = None, color: str = None, agg: str = "sum", bins: int = 30):
    """
    Return a plotly figure for common chart types.
    chart_type ∈ {"bar","line","scatter","hist","box"}
    """
    if chart_type == "bar":
        if x and y:
            if is_numeric(df, y):
                g = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
                return px.bar(g, x=x, y=y, color=color or None)
        # fallback: top categories of x
        if x and not y:
            g = df[x].astype(str).value_counts().head(20).reset_index(names=[x,"count"])
            return px.bar(g, x=x, y="count")
    elif chart_type == "line":
        if x and y:
            g = df.groupby(x, dropna=False)[y].agg(agg).reset_index()
            return px.line(g, x=x, y=y, color=color or None)
    elif chart_type == "scatter":
        if x and y and is_numeric(df, x) and is_numeric(df, y):
            return px.scatter(df, x=x, y=y, color=color or None)
    elif chart_type == "hist":
        if x:
            return px.histogram(df, x=x, nbins=bins, color=color or None)
    elif chart_type == "box":
        if x and y:
            return px.box(df, x=x, y=y, color=color or None)
        if y and not x:
            return px.box(df, y=y, color=color or None)
    return None

def corr_heatmap(df: pd.DataFrame):
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return None
    corr = num.corr(numeric_only=True)
    return px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap")
