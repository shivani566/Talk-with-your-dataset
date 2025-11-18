import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load a lightweight sentence transformer for semantic matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define scalable templates
TEMPLATES = {
    "aggregate": "df.groupby('{group_by}')[{column}].{agg}().reset_index()",
    "filter": "df[df['{column}'] {operator} {value}]",
    "sort": "df.sort_values(by='{column}', ascending={asc}).head({n})",
    "trend": "df.groupby('{time_col}')[{column}].sum().reset_index()",
    "missing": "df['{column}'].isnull().sum()",
    "correlation": "df['{col1}'].corr(df['{col2}'])"
}

# Example mapping of keywords to operation type
OP_KEYWORDS = {
    "sum": ("aggregate", "sum"),
    "average": ("aggregate", "mean"),
    "mean": ("aggregate", "mean"),
    "median": ("aggregate", "median"),
    "count": ("aggregate", "count"),
    "top": ("sort", "desc"),
    "bottom": ("sort", "asc"),
    "trend": ("trend", None),
    "filter": ("filter", None),
    "null": ("missing", None),
    "correlation": ("correlation", None)
}

def semantic_match_column(query, df_columns):
    """
    Return the best matching column from df_columns for a given query string.
    Uses cosine similarity from sentence-transformers.
    """
    query_emb = model.encode(query, convert_to_tensor=True)
    columns_emb = model.encode(df_columns, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, columns_emb)[0]
    best_idx = int(scores.argmax())
    return df_columns[best_idx]

def parse_query(query, df):
    """
    Parse the NL query and map it to operation type, columns, and values.
    """
    query_lower = query.lower()
    op_type = None
    agg = None
    operator = None
    column = None
    group_by = None
    time_col = None
    col2 = None
    n = None
    asc = True
    value = None

    # 1️⃣ Detect operation + agg
    for key, val in OP_KEYWORDS.items():
        if key in query_lower:
            op_type, agg_val = val
            if op_type == "aggregate":
                agg = agg_val
            if op_type == "sort":
                asc = True if agg_val == "asc" else False
            break

    # 2️⃣ Map column(s) using semantic match
    if df.columns.any():
        column = semantic_match_column(query, df.columns)
        if "by " in query_lower:
            # extract grouping column if exists
            for c in df.columns:
                if f"by {c.lower()}" in query_lower:
                    group_by = c
        # correlation second column
        if op_type == "correlation":
            for c in df.columns:
                if c != column:
                    col2 = c
                    break
        # filter value
        if op_type == "filter":
            # simple heuristic: pick a word after '=' or 'is'
            if "=" in query_lower:
                value = query_lower.split("=")[-1].strip().strip("'\"")
            elif "is" in query_lower:
                value = query_lower.split("is")[-1].strip().strip("'\"")
        # trend time column
        if op_type == "trend":
            time_col = semantic_match_column("date time column", df.select_dtypes(include=["datetime","object"]).columns)

    return {
        "op_type": op_type,
        "agg": agg,
        "column": column,
        "group_by": group_by,
        "operator": operator,
        "value": value,
        "n": n,
        "asc": asc,
        "time_col": time_col,
        "col2": col2
    }

def answer_query(df, query: str):
    """
    Handles NL queries over a dataframe using hybrid template + AI for column mapping.
    """
    try:
        parsed = parse_query(query, df)
        op_type = parsed["op_type"]

        if op_type not in TEMPLATES or not parsed["column"]:
            return {"error": "Sorry, I couldn't understand your query.", "result": None, "code": None, "summary": ""}

        # Fill template
        if op_type == "aggregate":
            code_to_run = TEMPLATES[op_type].format(
                column=f"'{parsed['column']}'",
                group_by=f"'{parsed['group_by']}'" if parsed["group_by"] else f"'{parsed['column']}'",
                agg=parsed["agg"]
            )
        elif op_type == "filter":
            code_to_run = TEMPLATES[op_type].format(
                column=f"'{parsed['column']}'",
                operator="==",
                value=f"'{parsed['value']}'"
            )
        elif op_type == "sort":
            code_to_run = TEMPLATES[op_type].format(
                column=f"'{parsed['column']}'",
                asc=parsed["asc"],
                n=5
            )
        elif op_type == "trend":
            code_to_run = TEMPLATES[op_type].format(
                time_col=f"'{parsed['time_col']}'",
                column=f"'{parsed['column']}'"
            )
        elif op_type == "missing":
            code_to_run = TEMPLATES[op_type].format(column=f"'{parsed['column']}'")
        elif op_type == "correlation":
            code_to_run = TEMPLATES[op_type].format(col1=f"'{parsed['column']}'", col2=f"'{parsed['col2']}'")
        else:
            return {"error": "Operation type not supported yet.", "result": None, "code": None, "summary": ""}

        # Execute code
        result = eval(code_to_run)

        # Prepare summary
        if isinstance(result, pd.DataFrame):
            summary = f"The query returned a DataFrame with {result.shape[0]} rows and {result.shape[1]} columns."
        elif isinstance(result, pd.Series):
            summary = f"The query returned a Series with {result.shape[0]} values."
        else:
            summary = str(result)

        return {"result": result, "code": code_to_run, "summary": summary}

    except Exception as e:
        return {"error": str(e), "result": None, "code": None, "summary": "Error occurred while processing query."}











