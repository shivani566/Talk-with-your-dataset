import pandas as pd
from io import BytesIO, StringIO

def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load CSV or Excel into a DataFrame from a Streamlit UploadedFile.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # handle encodings gracefully
        try:
            return pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            raw = uploaded_file.read()
            return pd.read_csv(StringIO(raw.decode("latin-1")))
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Only CSV and Excel (.xlsx/.xls) files are supported.")
