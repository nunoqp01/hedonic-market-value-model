import pandas as pd

RAW_PATH = "data/raw/Base.xlsx"
SHEET = "Data"


def fix_mojibake(text):
    """Repair UTF-8 text that was wrongly decoded as Windows-1252."""
    out = bytearray()
    for char in text:
        try:
            out += char.encode("cp1252")
        except UnicodeEncodeError:
            try:
                out += char.encode("latin-1")
            except UnicodeEncodeError:
                return text  # unexpected character: leave the text untouched
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return text  # text was not corrupted: leave it as it is


def load_data():
    """Read the raw players file and return a cleaned DataFrame."""
    df = pd.read_excel(RAW_PATH, sheet_name=SHEET)
    df = df.rename(columns={"npxg_p90 final": "npxg_p90"})
    df["name"] = df["name"].map(fix_mojibake)
    df["club"] = df["club"].map(fix_mojibake)
    return df