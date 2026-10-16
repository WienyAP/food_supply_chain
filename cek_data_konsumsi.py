
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

print("=== FILE DATA YANG TERSEDIA ===")
for file in DATA_DIR.iterdir():
    if file.is_file():
        print(file.name)

print("\n=== FILE CSV TERKAIT KONSUMSI ===")
for file in DATA_DIR.glob("*.csv"):
    if any(k in file.name.lower() for k in ["konsumsi", "consumption", "beras"]):
        df = pd.read_csv(file)

        print(f"\nFile: {file.name}")
        print("Kolom:", df.columns.tolist())
        print(df.head(3).to_string(index=False))
