"""Download the UCI Online Retail II dataset and convert it to CSV.

Source: Chen, D. (2012). Online Retail II. UCI Machine Learning Repository.
https://doi.org/10.24432/C5CG6D  (CC BY 4.0)
"""

from pathlib import Path

import pandas as pd
import requests

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00502/online_retail_II.xlsx"

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
XLSX_PATH = RAW_DIR / "online_retail_II.xlsx"
CSV_PATH = RAW_DIR / "online_retail_ii.csv"


def download() -> None:
    if XLSX_PATH.exists():
        print(f"Already downloaded: {XLSX_PATH}")
        return

    print(f"Downloading {URL} ...")
    response = requests.get(URL, stream=True, timeout=120)
    response.raise_for_status()

    with open(XLSX_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    size_mb = XLSX_PATH.stat().st_size / 1_048_576
    print(f"Downloaded {size_mb:.1f} MB")


def convert() -> None:
    print("Reading Excel (this takes 1-3 minutes)...")
    sheets = pd.read_excel(XLSX_PATH, sheet_name=None)

    print(f"Sheets found: {list(sheets.keys())}")
    for name, sheet in sheets.items():
        print(f"  {name}: {len(sheet):,} rows")

    df = pd.concat(sheets.values(), ignore_index=True)

    print(f"Combined: {len(df):,} rows")
    print(f"Columns: {list(df.columns)}")

    df.to_csv(CSV_PATH, index=False)
    print(f"Saved to {CSV_PATH}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download()
    convert()


if __name__ == "__main__":
    main()