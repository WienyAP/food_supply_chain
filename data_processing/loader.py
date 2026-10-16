
from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


def load_csv(filename: str) -> pd.DataFrame:
    """Membaca CSV dari folder data proyek."""
    file_path = DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"File tidak ditemukan: {file_path}\n"
            "Pastikan nama file dan lokasi folder data sudah benar."
        )

    return pd.read_csv(file_path)


def load_produksi() -> pd.DataFrame:
    return load_csv("Produksi Beras all 2020-2025.csv")


def load_konsumsi() -> pd.DataFrame:
    return load_csv("Konsumsi pangan KabKota 2021-2025.csv")


def load_penduduk() -> pd.DataFrame:
    return load_csv(
        "jumlah_penduduk_perkapita_2020-2026.csv"
    )


def load_distribusi() -> pd.DataFrame:
    return load_csv("distribusi_pangan_all.csv")
