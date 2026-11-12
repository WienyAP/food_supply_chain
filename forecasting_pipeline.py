
from pathlib import Path
import pandas as pd


# ============================================================
# 1. KONFIGURASI DIREKTORI
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"


# ============================================================
# 2. FUNGSI STANDARDISASI NAMA WILAYAH
# ============================================================

def norm_region(value):
    """
    Menyeragamkan nama kabupaten/kota agar dapat dicocokkan
    antar-dataset.
    """
    value = (
        " ".join(
            str(value)
            .strip()
            .casefold()
            .replace("kab. ", "kabupaten ")
            .split()
        )
    )

    if value.startswith("kabupaten "):
        value = value[len("kabupaten "):]

    return value


# ============================================================
# 3. FUNGSI VALIDASI KOLOM
# ============================================================

def validate_columns(df, required_columns, dataset_name):
    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Kolom pada {dataset_name} tidak lengkap: {missing}"
        )


# ============================================================
# 4. PIPELINE FORECASTING
# ============================================================

def generate_forecasts(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    forecast_year=2026
):

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # A. MEMBACA DATA
    # --------------------------------------------------------

    supply = pd.read_csv(data_dir / "monthly_supply.csv")

    pop = pd.read_csv(
        data_dir / "jumlah_penduduk_perkapita_2020-2026.csv"
    )

    consumption = pd.read_csv(
        data_dir / "Konsumsi pangan KabKota 2021-2025.csv"
    )

    print("\n=== DATASET YANG DIGUNAKAN ===")
    print("Supply:", supply.shape)
    print("Population:", pop.shape)
    print("Consumption:", consumption.shape)

    # --------------------------------------------------------
    # B. VALIDASI STRUKTUR DATA
    # --------------------------------------------------------

    validate_columns(
        supply,
        {
            "region_id",
            "region_standard",
            "year",
            "month",
            "production_ton"
        },
        "monthly_supply.csv"
    )

    validate_columns(
        pop,
        {
            "year",
            "region",
            "jumlah_penduduk"
        },
        "jumlah_penduduk_perkapita_2020-2026.csv"
    )

    validate_columns(
        consumption,
        {
            "year",
            "region",
            "comodity",
            "Konsumsi_Pangan"
        },
        "Konsumsi pangan KabKota 2021-2025.csv"
    )

    # --------------------------------------------------------
    # C. PREPROCESSING DATA SUPPLY
    # --------------------------------------------------------

    supply["year"] = pd.to_numeric(
        supply["year"], errors="raise"
    ).astype(int)

    supply["month"] = pd.to_numeric(
        supply["month"], errors="raise"
    ).astype(int)

    supply["production_ton"] = pd.to_numeric(
        supply["production_ton"], errors="coerce"
    )

    hist = supply[
        supply["year"] == forecast_year - 1
    ].copy()

    if hist.empty:
        raise ValueError(
            f"Data supply tahun {forecast_year - 1} tidak ditemukan."
        )

    # Pastikan seluruh wilayah memiliki 12 bulan data.
    counts = hist.groupby("region_standard")["month"].nunique()

    incomplete = counts[counts != 12]

    if not incomplete.empty:
        raise ValueError(
            "Data produksi tidak lengkap 12 bulan untuk wilayah: "
            + str(incomplete.to_dict())
        )

    # Periksa duplikasi.
    if hist.duplicated(
        ["region_standard", "month"]
    ).any():

        raise ValueError(
            "Terdapat duplikasi wilayah-bulan pada data supply."
        )

    # --------------------------------------------------------
    # D. FORECAST SUPPLY: SEASONAL NAIVE
    # --------------------------------------------------------

    # Seasonal Naive:
    # Produksi bulan yang sama pada tahun sebelumnya
    # digunakan sebagai prediksi tahun berikutnya.

    forecast = hist[
        [
            "region_id",
            "region_standard",
            "month",
            "production_ton"
        ]
    ].copy()

    forecast = forecast.rename(
        columns={
            "production_ton": "forecast_supply_ton"
        }
    )

    forecast["year"] = forecast_year

    forecast["period"] = forecast.apply(
        lambda row: (
            f"{int(row['year']):04d}-{int(row['month']):02d}"
        ),
        axis=1
    )

    forecast["forecast_method"] = (
        "Seasonal Naive (same month previous year)"
    )

    forecast["forecast_origin"] = (
        f"{forecast_year - 1}-12"
    )

    forecast["source_period"] = forecast.apply(
        lambda row: (
            f"{forecast_year - 1}-{int(row['month']):02d}"
        ),
        axis=1
    )

    forecast["source_type"] = "forecast"

    # --------------------------------------------------------
    # E. PREPROCESSING DATA POPULASI
    # --------------------------------------------------------

    pop["year"] = pd.to_numeric(
        pop["year"], errors="raise"
    ).astype(int)

    pop26 = pop[
        pop["year"] == forecast_year
    ].copy()

    if pop26.empty:
        raise ValueError(
            f"Data penduduk tahun {forecast_year} tidak ditemukan."
        )

    pop26["jumlah_penduduk"] = pd.to_numeric(
        pop26["jumlah_penduduk"],
        errors="raise"
    )

    pop26["region_key"] = pop26["region"].map(
        norm_region
    )

    if pop26["region_key"].duplicated().any():
        raise ValueError(
            "Terdapat duplikasi wilayah pada data penduduk."
        )

    # --------------------------------------------------------
    # F. PREPROCESSING DATA KONSUMSI
    # --------------------------------------------------------

    consumption["year"] = pd.to_numeric(
        consumption["year"], errors="raise"
    ).astype(int)

    # Ambil hanya data konsumsi beras tahun 2025.
    cons_latest = consumption[
        (
            consumption["year"] == forecast_year - 1
        )
        &
        (
            consumption["comodity"]
            .astype(str)
            .str.strip()
            .str.casefold() == "beras"
        )
    ].copy()

    if cons_latest.empty:
        raise ValueError(
            "Data konsumsi beras tahun 2025 tidak ditemukan."
        )

    # Standardisasi nama kolom konsumsi.
    cons_latest = cons_latest.rename(
        columns={
            "Konsumsi_Pangan": "konsumsi_beras_per_kapita"
        }
    )

    cons_latest["konsumsi_beras_per_kapita"] = pd.to_numeric(
        cons_latest["konsumsi_beras_per_kapita"],
        errors="raise"
    )

    cons_latest["region_key"] = cons_latest["region"].map(
        norm_region
    )

    if cons_latest["region_key"].duplicated().any():
        raise ValueError(
            "Terdapat duplikasi wilayah pada data konsumsi beras."
        )

    # --------------------------------------------------------
    # G. MENGGABUNGKAN DATA POPULASI DAN KONSUMSI
    # --------------------------------------------------------

    demand_base = pop26.merge(
        cons_latest[
            [
                "region_key",
                "konsumsi_beras_per_kapita"
            ]
        ],
        on="region_key",
        how="outer",
        indicator=True
    )

    unmatched = demand_base[
        demand_base["_merge"] != "both"
    ]

    if not unmatched.empty:
        raise ValueError(
            "Nama wilayah penduduk dan konsumsi tidak cocok: "
            + str(
                unmatched[
                    ["region_key", "_merge"]
                ].to_dict("records")
            )
        )

    demand_base = demand_base.drop(
        columns=["_merge"]
    )

    # --------------------------------------------------------
    # H. MENGHITUNG ESTIMASI DEMAND
    # --------------------------------------------------------

    # Konsumsi per kapita menggunakan satuan kg/orang/tahun.
    # Populasi x konsumsi / 1000 menghasilkan ton/tahun.

    demand_base["annual_demand_ton"] = (
        demand_base["jumlah_penduduk"]
        *
        demand_base["konsumsi_beras_per_kapita"]
        / 1000
    )

    # Alokasi konsumsi tahunan secara merata ke 12 bulan.
    demand_base["estimated_demand_ton"] = (
        demand_base["annual_demand_ton"] / 12
    )

    # --------------------------------------------------------
    # I. MENGGABUNGKAN DEMAND DENGAN FORECAST SUPPLY
    # --------------------------------------------------------

    forecast["region_key"] = forecast[
        "region_standard"
    ].map(norm_region)

    demand_base = demand_base.set_index(
        "region_key"
    )

    forecast["population_2026"] = forecast[
        "region_key"
    ].map(
        demand_base["jumlah_penduduk"]
    )

    forecast["consumption_kg_per_capita_year"] = forecast[
        "region_key"
    ].map(
        demand_base["konsumsi_beras_per_kapita"]
    )

    forecast["estimated_demand_ton"] = forecast[
        "region_key"
    ].map(
        demand_base["estimated_demand_ton"]
    )

    # Validasi hasil mapping.
    demand_columns = [
        "population_2026",
        "consumption_kg_per_capita_year",
        "estimated_demand_ton"
    ]

    if forecast[demand_columns].isna().any().any():

        bad = forecast.loc[
            forecast["estimated_demand_ton"].isna(),
            "region_standard"
        ].unique().tolist()

        raise ValueError(
            "Wilayah tidak terpetakan ke data demand: "
            + str(bad)
        )

    # --------------------------------------------------------
    # J. MENGHITUNG BALANCE INDICATOR
    # --------------------------------------------------------

    forecast["balance_indicator_ton"] = (
        forecast["forecast_supply_ton"]
        -
        forecast["estimated_demand_ton"]
    )

    forecast["balance_status"] = forecast[
        "balance_indicator_ton"
    ].apply(
        lambda value: (
            "surplus_indicator"
            if value >= 0
            else "deficit_indicator"
        )
    )

    forecast["demand_method"] = (
        "2026 population x 2025 rice consumption per capita / 12 "
        "(uniform monthly allocation)"
    )

    forecast["interpretation_note"] = (
        "Proyeksi produksi Seasonal Naive dikurangi estimasi "
        "konsumsi bulanan; bukan stok aktual atau volume "
        "yang pasti dapat didistribusikan."
    )

    forecast = forecast.drop(
        columns=["region_key"]
    )

    # --------------------------------------------------------
    # K. MENYUSUN URUTAN KOLOM OUTPUT
    # --------------------------------------------------------

    output_columns = [
        "region_id",
        "region_standard",
        "year",
        "month",
        "period",
        "forecast_supply_ton",
        "estimated_demand_ton",
        "balance_indicator_ton",
        "balance_status",
        "population_2026",
        "consumption_kg_per_capita_year",
        "forecast_method",
        "forecast_origin",
        "source_period",
        "source_type",
        "demand_method",
        "interpretation_note"
    ]

    forecast = forecast[output_columns]

    forecast = forecast.sort_values(
        ["region_standard", "month"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # L. VALIDASI OUTPUT
    # --------------------------------------------------------

    expected_rows = forecast[
        "region_standard"
    ].nunique() * 12

    if len(forecast) != expected_rows:
        raise ValueError(
            "Jumlah baris output tidak sesuai dengan "
            "jumlah wilayah x 12 bulan."
        )

    if forecast.duplicated(
        ["region_standard", "year", "month"]
    ).any():
        raise ValueError(
            "Terdapat duplikasi wilayah-bulan pada hasil forecast."
        )

    # --------------------------------------------------------
    # M. MENYIMPAN OUTPUT
    # --------------------------------------------------------

    supply_output = output_dir / (
        "forecast_2026_seasonal_naive.csv"
    )

    balance_output = output_dir / (
        "forecast_supply_demand_balance_2026.csv"
    )

    # File pertama: proyeksi supply.
    forecast[
        [
            "region_id",
            "region_standard",
            "year",
            "month",
            "period",
            "forecast_supply_ton",
            "forecast_method",
            "forecast_origin",
            "source_period",
            "source_type"
        ]
    ].to_csv(
        supply_output,
        index=False
    )

    # File kedua: supply, demand, dan balance indicator.
    forecast.to_csv(
        balance_output,
        index=False
    )

    # --------------------------------------------------------
    # N. RINGKASAN HASIL
    # --------------------------------------------------------

    print("\n=== FORECAST BERHASIL ===")

    print(
        f"Jumlah baris: {len(forecast)}"
    )

    print(
        f"Jumlah wilayah: "
        f"{forecast['region_standard'].nunique()}"
    )

    print(
        f"Periode: "
        f"{forecast['period'].min()} "
        f"s.d. {forecast['period'].max()}"
    )

    print(
        "Total forecast supply:",
        round(
            forecast["forecast_supply_ton"].sum(),
            2
        ),
        "ton"
    )

    print(
        "Total estimated demand:",
        round(
            forecast["estimated_demand_ton"].sum(),
            2
        ),
        "ton"
    )

    print(
        "Net balance indicator:",
        round(
            forecast["balance_indicator_ton"].sum(),
            2
        ),
        "ton"
    )

    print(
        "Status balance:",
        forecast["balance_status"].value_counts().to_dict()
    )

    print("\nFile berhasil disimpan:")
    print(supply_output)
    print(balance_output)

    return forecast


# ============================================================
# 5. EKSEKUSI PROGRAM
# ============================================================

if __name__ == "__main__":

    df = generate_forecasts()

    print("\n=== PREVIEW HASIL ===")

    print(
        df[
            [
                "region_standard",
                "period",
                "forecast_supply_ton",
                "estimated_demand_ton",
                "balance_indicator_ton",
                "balance_status"
            ]
        ].head(15).to_string(index=False)
    )
