
import re
import pandas as pd


def normalize_region(value) -> str:
    """Normalisasi nama wilayah dengan mempertahankan identitas kabupaten/kota."""
    name = str(value).strip().lower()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"^kab\.\s*", "kabupaten ", name)
    name = re.sub(r"^kota administrasi\s+", "kota ", name)
    return name


def normalize_kabkota_region(value) -> str:
    """
    Dataset konsumsi dan penduduk menulis nama kabupaten tanpa awalan
    'Kabupaten', tetapi nama kota menggunakan awalan 'Kota'.
    """
    name = normalize_region(value)

    if name.startswith(("kabupaten ", "kota ")):
        return name

    return f"kabupaten {name}"


def prepare_production(df: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "month", "region", "produksi_beras"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom produksi tidak ditemukan: {sorted(missing)}")

    result = df[
        ["year", "month", "region", "produksi_beras"]
    ].copy()

    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result["month"] = pd.to_numeric(result["month"], errors="coerce")
    result["produksi_beras"] = pd.to_numeric(
        result["produksi_beras"], errors="coerce"
    )
    result["region"] = result["region"].astype("string").str.strip()

    if result[
        ["year", "month", "region", "produksi_beras"]
    ].isna().any().any():
        raise ValueError("Data produksi memiliki nilai kosong atau tidak valid.")

    result["year"] = result["year"].astype(int)
    result["month"] = result["month"].astype(int)

    if not result["month"].between(1, 12).all():
        raise ValueError("Nilai month harus berada pada rentang 1–12.")

    if (result["produksi_beras"] < 0).any():
        raise ValueError("Produksi beras tidak boleh negatif.")

    result["region_key"] = result["region"].map(normalize_region)

    if result.duplicated(["year", "month", "region_key"]).any():
        raise ValueError(
            "Ditemukan duplikasi wilayah-tahun-bulan pada produksi."
        )

    return result.sort_values(
        ["year", "region_key", "month"]
    ).reset_index(drop=True)


def prepare_consumption(df: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "region", "comodity", "Konsumsi_Pangan"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom konsumsi tidak ditemukan: {sorted(missing)}")

    result = df.copy()
    result = result[
        result["comodity"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("beras")
    ].copy()

    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result["Konsumsi_Pangan"] = pd.to_numeric(
        result["Konsumsi_Pangan"], errors="coerce"
    )
    result["region"] = result["region"].astype("string").str.strip()

    if result[
        ["year", "region", "Konsumsi_Pangan"]
    ].isna().any().any():
        raise ValueError(
            "Data konsumsi beras memiliki nilai kosong atau tidak valid."
        )

    result["year"] = result["year"].astype(int)

    if (result["Konsumsi_Pangan"] < 0).any():
        raise ValueError("Nilai konsumsi tidak boleh negatif.")

    result["region_key"] = result["region"].map(normalize_kabkota_region)

    if result.duplicated(["year", "region_key"]).any():
        raise ValueError(
            "Ditemukan duplikasi wilayah-tahun pada konsumsi beras."
        )

    return result.sort_values(
        ["year", "region_key"]
    ).reset_index(drop=True)


def prepare_population(df: pd.DataFrame) -> pd.DataFrame:
    required = {"year", "region", "jumlah_penduduk"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Kolom penduduk tidak ditemukan: {sorted(missing)}")

    result = df[
        ["year", "region", "jumlah_penduduk"]
    ].copy()

    result["year"] = pd.to_numeric(result["year"], errors="coerce")
    result["jumlah_penduduk"] = pd.to_numeric(
        result["jumlah_penduduk"], errors="coerce"
    )
    result["region"] = result["region"].astype("string").str.strip()

    if result[
        ["year", "region", "jumlah_penduduk"]
    ].isna().any().any():
        raise ValueError(
            "Data penduduk memiliki nilai kosong atau tidak valid."
        )

    result["year"] = result["year"].astype(int)

    if (result["jumlah_penduduk"] <= 0).any():
        raise ValueError("Jumlah penduduk harus lebih besar dari nol.")

    result["region_key"] = result["region"].map(normalize_kabkota_region)

    if result.duplicated(["year", "region_key"]).any():
        raise ValueError(
            "Ditemukan duplikasi wilayah-tahun pada data penduduk."
        )

    return result.sort_values(
        ["year", "region_key"]
    ).reset_index(drop=True)
