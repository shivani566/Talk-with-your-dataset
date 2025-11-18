import sys
print(sys.executable)
print(sys.path)

import streamlit as st
import pandas as pd
import logging
from src import (
    load_dataset, try_parse_dates, dataset_overview, summary_markdown,
    missing_summary, numeric_summary, categorical_summary,
    build_chart, corr_heatmap, guess_date_column, answer_query
)

st.set_page_config(page_title="Talk with your Dataset (DA/DS Edition)", layout="wide")
st.title("📊 Talk with your Dataset")
st.caption("Upload CSV/Excel → Auto summary → Ask questions → See trends & visuals")

# ----------------------------
# Sidebar - Upload & Help
# ----------------------------
with st.sidebar:
    st.subheader("Upload")
    file = st.file_uploader("CSV or Excel", type=["csv","xlsx","xls"])
    st.markdown("---")
    st.subheader("Help")
    st.markdown(
        """
**Sample queries**
- `sum of Sales by Region`
- `average of Profit by Category`
- `count by Segment`
- `top 5 products by Sales`
- `correlation between Discount and Profit`
- `trend of Sales over OrderDate`
- `filter Region = West`
- `show nulls in Profit`
        """
    )

if not file:
    st.info("⬅️ Upload a CSV/Excel to begin.")
    st.stop()

# ----------------------------
# Load Dataset
# ----------------------------
df = load_dataset(file)
df = try_parse_dates(df.copy())

st.subheader("👀 Preview")
st.dataframe(df.head(10), use_container_width=True)

# ----------------------------
# Dataset Overview
# ----------------------------
info = dataset_overview(df)
left, right = st.columns([1, 2])
with left:
    st.markdown("### 🧾 Overview")
    st.markdown(summary_markdown(info))
with right:
    st.markdown("### 🧮 Dtypes")
    st.dataframe(pd.DataFrame({"dtype": df.dtypes.astype(str)}), use_container_width=True)

# ----------------------------
# Missing / Numeric / Categorical Summaries
# ----------------------------
with st.expander("🧩 Missing Values"):
    ms = missing_summary(df)
    if ms.empty:
        st.success("No missing values detected 🎉")
    else:
        st.dataframe(ms, use_container_width=True)

with st.expander("📈 Numeric Summary (describe)"):
    ns = numeric_summary(df)
    if ns.empty:
        st.info("No numeric columns detected.")
    else:
        st.dataframe(ns, use_container_width=True)

with st.expander("🏷️ Categorical Snapshot (top 5 values per column)"):
    cs = categorical_summary(df)
    if cs.empty:
        st.info("No categorical columns detected.")
    else:
        st.dataframe(cs, use_container_width=True)

st.markdown("---")

# ----------------------------
# ----------------------------
# Natural Language Query
# ----------------------------
st.subheader("💬 Ask a question (Natural Language)")
query = st.text_input("Try: 'sum of Sales by Region' or 'trend of Sales over OrderDate'")

if query:
    with st.spinner("Processing your query..."):
        res = answer_query(df, query)

    if "error" in res:
        st.error(res["error"])
    else:
        st.subheader("📊 Answer")
        st.write(res.get("summary", "No summary available."))

        result = res.get("result")
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        elif isinstance(result, pd.Series):
            st.dataframe(result.to_frame(), use_container_width=True)
        else:
            st.write(result)

        st.subheader("🔎 Generated Pandas Code")
        st.code(res.get("code", "No code generated."))

        # Optional quick chart if small table
        try:
            if isinstance(result, pd.DataFrame) and result.shape[0] <= 100 and result.shape[1] <= 4:
                st.subheader("📈 Quick Chart of Result")
                # Use first column as index for chart
                st.bar_chart(result.set_index(result.columns[0]))
        except Exception:
            pass

# ----------------------------
# Automatic Chart for Query Result
# ----------------------------
if query:
    result = res.get("result")
    if isinstance(result, pd.DataFrame) and not result.empty:
        numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            st.subheader("📊 Automatic Chart")
            # Choose first numeric column for chart
            y_col = numeric_cols[0]
            # Use first non-numeric column as x-axis if exists
            x_col = None
            for c in result.columns:
                if c not in numeric_cols:
                    x_col = c
                    break
            if x_col:
                st.bar_chart(result.set_index(x_col)[y_col])
            else:
                st.line_chart(result[y_col])
        else:
            st.info("No numeric columns found in the result to plot a chart.")

# ----------------------------
# Optional: Suggest Related Queries
# ----------------------------
st.markdown("---")
st.subheader("💡 Suggestions")
st.markdown(
    """
- Try asking for `average of Profit by Category`
- Check `top 5 Sales by Region`
- Find `count by Segment`
- Explore `nulls in Discount`
- See `correlation between Profit and Sales`
"""
)


# ----------------------------
# Manual Charts
# ----------------------------
st.markdown("---")
st.subheader("📊 Build a Chart")
chart_type = st.selectbox("Chart type", ["bar","line","scatter","hist","box"])
cols = list(df.columns)
x = st.selectbox("X", [None] + cols, index=0)
y = st.selectbox("Y", [None] + cols, index=0)
color = st.selectbox("Color (optional)", [None] + cols, index=0)
agg = st.selectbox("Aggregation (for bar/line)", ["sum","mean","median","count"], index=0)
bins = st.slider("Bins (histogram only)", 5, 100, 30)

fig = build_chart(df, chart_type, x=x, y=y, color=color, agg=agg, bins=bins)
if fig:
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pick appropriate X/Y for the selected chart.")

# ----------------------------
# Correlation Heatmap
# ----------------------------
with st.expander("🔗 Correlation Heatmap"):
    h = corr_heatmap(df)
    if h:
        st.plotly_chart(h, use_container_width=True)
    else:
        st.info("Need ≥ 2 numeric columns.")

st.markdown("---")
st.caption("Tip: keep column names clear (e.g., 'OrderDate', 'Sales', 'Region') for better NL matching.")

