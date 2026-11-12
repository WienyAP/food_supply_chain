from pathlib import Path
import re
import pandas as pd

# Jalankan dari root project food_supply_chain.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "production": DATA_DIR / "Produksi Beras all 2020-2025.csv",
    "consumption": DATA_DIR / "Konsumsi pangan KabKota 2021-2025.csv",
    "population": DATA_DIR / "jumlah_penduduk_perkapita_2020-2026.csv",
    "climate": DATA_DIR / "jatim_climate_monthly (1).csv",
    "regions": DATA_DIR / "region_master.csv",
}


def normalize_region(value):
    """Normalisasi nama wilayah untuk pencocokan lintas dataset."""
    value = str(value).strip().lower()
    value = re.sub(r"\bkabupaten\b|\bkab\.?\s*", "", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def require_columns(df, columns, label):
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Kolom {missing} tidak ditemukan pada {label}.")

def load_inputs():
    for label, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"File {label} tidak ditemukan: {path}\n"
                "Pastikan file berada di dalam folder data/ dengan nama yang sesuai."
            )

    return {
        "production": pd.read_csv(FILES["production"]),
        "consumption": pd.read_csv(FILES["consumption"]),
        "population": pd.read_csv(FILES["population"]),
        "climate": pd.read_csv(FILES["climate"]),
        "regions": pd.read_csv(FILES["regions"]),
    }


def build_region_lookup(regions):
    require_columns(
        regions,
        ["region_id", "region_standard", "production_region", "demand_region"],
        "region_master.csv",
    )
    lookup = {}
    for row in regions.itertuples(index=False):
        for candidate in (row.region_standard, row.production_region, row.demand_region):
            key = normalize_region(candidate)
            if key:
                if key in lookup and lookup[key]["region_id"] != row.region_id:
                    raise ValueError(f"Nama wilayah ambigu pada region_master: {candidate}")
                lookup[key] = {
                    "region_id": row.region_id,
                    "region": row.region_standard,
                }
    return lookup

def map_regions(df, source_col, lookup, label):
    result = df.copy()
    result["_region_key"] = result[source_col].map(normalize_region)
    result["region_id"] = result["_region_key"].map(
        lambda key: lookup.get(key, {}).get("region_id")
    )
    result["region_standard"] = result["_region_key"].map(
        lambda key: lookup.get(key, {}).get("region")
    )
    unmatched = sorted(result.loc[result["region_id"].isna(), source_col].astype(str).unique())
    if unmatched:
        raise ValueError(
            f"Ada nama wilayah {label} yang tidak cocok dengan region_master: {unmatched}"
        )
    return result

def build_monthly_supply(data, lookup):
    df = data["production"].copy()
    require_columns(df, ["year", "month", "region", "produksi_beras"], "data produksi")
    df = map_regions(df, "region", lookup, "produksi")
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)
    df["production_ton"] = pd.to_numeric(df["produksi_beras"], errors="coerce")

    if df["production_ton"].isna().any():
        raise ValueError("Ada nilai produksi_beras yang kosong atau bukan angka.")
    if not df["month"].between(1, 12).all():
        raise ValueError("Kolom month harus berada pada rentang 1 sampai 12.")

    keys = ["region_id", "year", "month"]
    duplicates = df.duplicated(keys, keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, keys].head().to_dict("records")
        raise ValueError(f"Duplikasi data produksi ditemukan. Contoh: {sample}")

    # Gabungkan data iklim bulanan berdasarkan ID wilayah dan periode.
    climate = data["climate"].copy()
    require_columns(
        climate,
        ["region", "period", "rainfall_mm_month", "temp_avg_c", "humidity_pct", "solar_mj"],
        "data iklim",
    )
    climate = map_regions(climate, "region", lookup, "iklim")
    climate["period"] = climate["period"].astype(str)
    climate["year"] = climate["period"].str.slice(0, 4).astype(int)
    climate["month"] = climate["period"].str.slice(5, 7).astype(int)
    climate_cols = [
        "region_id", "year", "month", "rainfall_mm_month",
        "temp_avg_c", "humidity_pct", "solar_mj",
    ]
    climate = climate[climate_cols].copy()
    if climate.duplicated(["region_id", "year", "month"]).any():
        raise ValueError("Duplikasi wilayah-bulan ditemukan pada data iklim.")

    df = df.merge(climate, on=["region_id", "year", "month"], how="left", validate="one_to_one")
    df["source_type"] = "historical_observation"
    df["production_unit"] = "ton"
    df["period"] = df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)

    columns = [
        "region_id", "region_standard", "year", "month", "period",
        "production_ton", "rainfall_mm_month", "temp_avg_c",
        "humidity_pct", "solar_mj", "source_type", "production_unit",
    ]
    return df[columns].sort_values(["region_id", "year", "month"]).reset_index(drop=True)

def build_monthly_demand(data, lookup):
    consumption = data["consumption"].copy()
    population = data["population"].copy()
    require_columns(consumption, ["year", "region", "comodity", "Konsumsi_Pangan"], "data konsumsi")
    require_columns(population, ["year", "region", "jumlah_penduduk"], "data penduduk")

    consumption = consumption[consumption["comodity"].astype(str).str.strip().str.lower() == "beras"].copy()
    consumption = map_regions(consumption, "region", lookup, "konsumsi")
    population = map_regions(population, "region", lookup, "penduduk")

    for df in (consumption, population):
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)

    consumption["consumption_kg_per_capita_year"] = pd.to_numeric(
        consumption["Konsumsi_Pangan"], errors="coerce"
    )
    population["population"] = pd.to_numeric(population["jumlah_penduduk"], errors="coerce")
    if consumption["consumption_kg_per_capita_year"].isna().any():
        raise ValueError("Ada nilai konsumsi yang kosong atau bukan angka.")
    if population["population"].isna().any():
        raise ValueError("Ada nilai jumlah penduduk yang kosong atau bukan angka.")

    keys = ["region_id", "year"]
    if consumption.duplicated(keys).any() or population.duplicated(keys).any():
        raise ValueError("Duplikasi wilayah-tahun ditemukan pada konsumsi atau penduduk.")

    annual = consumption[keys + ["region_standard", "consumption_kg_per_capita_year"]].merge(
        population[keys + ["population"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if annual.empty:
        raise ValueError("Tidak ada irisan tahun-wilayah antara konsumsi dan penduduk.")

    annual["annual_demand_ton"] = (
        annual["population"] * annual["consumption_kg_per_capita_year"] / 1000
    )
    annual["estimated_monthly_demand_ton"] = annual["annual_demand_ton"] / 12
    annual["source_type"] = "estimated_from_annual_per_capita"
    annual["demand_method"] = "uniform_monthly_allocation"

    monthly = annual.loc[annual.index.repeat(12)].copy()
    monthly["month"] = monthly.groupby(keys).cumcount() + 1
    monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    monthly["demand_ton"] = monthly["estimated_monthly_demand_ton"]

    columns = [
        "region_id", "region_standard", "year", "month", "period",
        "population", "consumption_kg_per_capita_year", "annual_demand_ton",
        "demand_ton", "source_type", "demand_method",
    ]
    return monthly[columns].sort_values(["region_id", "year", "month"]).reset_index(drop=True)


def build_monthly_balance(supply, demand):
    keys = ["region_id", "year", "month", "period"]
    balance = supply.merge(
        demand,
        on=keys,
        how="inner",
        suffixes=("_supply", "_demand"),
        validate="one_to_one",
    )
    balance["region"] = balance["region_standard_supply"]
    balance["balance_ton"] = balance["production_ton"] - balance["demand_ton"]
    balance["balance_status"] = "balanced"
    balance.loc[balance["balance_ton"] > 0, "balance_status"] = "surplus_indicator"
    balance.loc[balance["balance_ton"] < 0, "balance_status"] = "deficit_indicator"
    balance["analysis_type"] = "historical_production_vs_estimated_demand"
    balance["interpretation_note"] = (
        "Indikator produksi dikurangi estimasi konsumsi; bukan stok aktual atau volume yang pasti dapat dikirim."
    )

    columns = [
        "region_id", "region", "year", "month", "period",
        "production_ton", "demand_ton", "balance_ton", "balance_status",
        "rainfall_mm_month", "temp_avg_c", "humidity_pct", "solar_mj",
        "source_type_demand", "demand_method", "analysis_type", "interpretation_note",
    ]
    # source_type_demand berasal dari tabel demand; hindari ketergantungan pada kolom lain.
    balance = balance.rename(columns={"source_type": "source_type_supply"})
    balance["source_type_demand"] = "estimated_from_annual_per_capita"
    return balance[columns].sort_values(["region_id", "year", "month"]).reset_index(drop=True)

def main():
    data = load_inputs()
    lookup = build_region_lookup(data["regions"])

    supply = build_monthly_supply(data, lookup)
    demand = build_monthly_demand(data, lookup)
    balance = build_monthly_balance(supply, demand)

    supply_path = OUTPUT_DIR / "monthly_supply.csv"
    demand_path = OUTPUT_DIR / "monthly_demand.csv"
    balance_path = OUTPUT_DIR / "monthly_balance.csv"

    supply.to_csv(supply_path, index=False)
    demand.to_csv(demand_path, index=False)
    balance.to_csv(balance_path, index=False)

    print("Pipeline berhasil dijalankan.")
    print(f"monthly_supply   : {len(supply):,} baris -> {supply_path}")
    print(f"monthly_demand   : {len(demand):,} baris -> {demand_path}")
    print(f"monthly_balance  : {len(balance):,} baris -> {balance_path}")
    print("Periode supply  :", supply["period"].min(), "s.d.", supply["period"].max())
    print("Periode demand  :", demand["period"].min(), "s.d.", demand["period"].max())
    print("Periode balance :", balance["period"].min(), "s.d.", balance["period"].max())
    print("\nDistribusi status balance:")
    print(balance["balance_status"].value_counts().to_string())
    print("\nContoh hasil:")
    print(balance[["region", "period", "production_ton", "demand_ton", "balance_ton", "balance_status"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
