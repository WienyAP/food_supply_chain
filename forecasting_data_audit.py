from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUPPLY_PATH = ROOT / "outputs" / "monthly_supply.csv"
DEMAND_PATH = ROOT / "outputs" / "monthly_demand.csv"
CLIMATE_PATH = ROOT / "data" / "jatim_climate_monthly (1).csv"
if not CLIMATE_PATH.exists():
    CLIMATE_PATH = ROOT / "jatim_climate_monthly (1).csv"
OUT = ROOT / "outputs" / "forecasting_audit"

CLIMATE_FEATURES = [
    "rainfall_mm_month", "temp_avg_c", "humidity_pct", "solar_mj"
]

def require_columns(df, columns, label):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{label}: kolom wajib tidak ditemukan: {missing}")

def audit_panel(df, label, value_col):
    require_columns(df, ["region_standard", "period", value_col], label)
    x = df.copy()
    x["period_dt"] = pd.to_datetime(x["period"] + "-01", errors="coerce")
    if x["period_dt"].isna().any():
        raise ValueError(f"{label}: ada nilai period yang tidak valid.")
    duplicates = int(x.duplicated(["region_standard", "period"]).sum())
    rows = []
    for region, g in x.groupby("region_standard"):
        g = g.sort_values("period_dt")
        full = pd.date_range(g["period_dt"].min(), g["period_dt"].max(), freq="MS")
        observed = pd.DatetimeIndex(g["period_dt"])
        missing_periods = full.difference(observed)
        vals = pd.to_numeric(g[value_col], errors="coerce")
        rows.append({
            "dataset": label,
            "region": region,
            "n_rows": len(g),
            "start_period": g["period"].min(),
            "end_period": g["period"].max(),
            "missing_value_count": int(vals.isna().sum()),
            "zero_value_count": int((vals == 0).sum()),
            "negative_value_count": int((vals < 0).sum()),
            "missing_month_count": len(missing_periods),
            "duplicate_region_period_count": int(g.duplicated(["region_standard", "period"]).sum()),
        })
    return pd.DataFrame(rows), duplicates

def main():
    for p in [SUPPLY_PATH, DEMAND_PATH, CLIMATE_PATH]:
        if not p.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {p}\n"
                "Pastikan monthly_supply.csv dan monthly_demand.csv berada di outputs/, "
                "serta file iklim berada di data/ atau root project."
            )

    supply = pd.read_csv(SUPPLY_PATH)
    demand = pd.read_csv(DEMAND_PATH)
    climate = pd.read_csv(CLIMATE_PATH)

    supply_audit, supply_dups = audit_panel(supply, "supply", "production_ton")
    demand_audit, demand_dups = audit_panel(demand, "demand", "demand_ton")

    require_columns(climate, ["region", "period", *CLIMATE_FEATURES], "climate")
    climate["period_dt"] = pd.to_datetime(climate["period"] + "-01", errors="coerce")
    if climate["period_dt"].isna().any():
        raise ValueError("Climate: ada nilai period yang tidak valid.")
    climate_missing = climate[
        climate[CLIMATE_FEATURES].isna().any(axis=1)
    ].copy()
    climate_hist = climate[climate["period_dt"] <= pd.Timestamp("2025-12-01")].copy()
    climate_hist_missing = climate_hist[
        climate_hist[CLIMATE_FEATURES].isna().any(axis=1)
    ].copy()

    zero_supply = supply[pd.to_numeric(supply["production_ton"], errors="coerce") == 0].copy()

    # Demand monthly series may be repeated from annual per-capita estimates.
    demand_variability = (
        demand.groupby(["region_standard", "year"], as_index=False)
        .agg(
            months=("period", "nunique"),
            unique_monthly_demand=("demand_ton", "nunique"),
            min_monthly_demand=("demand_ton", "min"),
            max_monthly_demand=("demand_ton", "max"),
        )
    )

    # Verify that climate history has one row per region-month and covers supply history.
    climate_dups = int(climate.duplicated(["region", "period"]).sum())
    climate_by_year = (
        climate.assign(year=climate["period"].str[:4].astype(int))
        .groupby("year")[CLIMATE_FEATURES]
        .agg(lambda col: int(col.isna().sum()))
        .reset_index()
    )

    OUT.mkdir(parents=True, exist_ok=True)
    quality = pd.concat([supply_audit, demand_audit], ignore_index=True)
    quality.to_csv(OUT / "data_quality_by_region.csv", index=False)
    zero_supply.to_csv(OUT / "zero_production_records.csv", index=False)
    climate_missing.to_csv(OUT / "climate_missing_records.csv", index=False)
    demand_variability.to_csv(OUT / "demand_variability_by_region_year.csv", index=False)
    climate_by_year.to_csv(OUT / "climate_missing_by_year.csv", index=False)

    report = [
        "# Forecasting Data Audit",
        "",
        "## Ringkasan panel",
        f"- Supply: {len(supply):,} baris; {supply['region_standard'].nunique()} wilayah; "
        f"{supply['period'].min()} s.d. {supply['period'].max()}; "
        f"duplikasi region-period: {supply_dups}.",
        f"- Demand: {len(demand):,} baris; {demand['region_standard'].nunique()} wilayah; "
        f"{demand['period'].min()} s.d. {demand['period'].max()}; "
        f"duplikasi region-period: {demand_dups}.",
        f"- Climate: {len(climate):,} baris; {climate['region'].nunique()} wilayah; "
        f"{climate['period'].min()} s.d. {climate['period'].max()}; "
        f"duplikasi region-period: {climate_dups}.",
        "",
        "## Temuan kualitas",
        f"- Produksi nol: {len(zero_supply):,} observasi. Nilai nol dipertahankan; "
        "perlu ditelusuri apakah nol aktual atau kode data tidak tersedia.",
        f"- Produksi negatif: {(pd.to_numeric(supply['production_ton'], errors='coerce') < 0).sum():,}.",
        f"- Demand bulanan: {demand['demand_ton'].isna().sum():,} nilai kosong.",
        f"- Baris iklim yang memiliki setidaknya satu fitur kosong: {len(climate_missing):,}.",
        f"- Baris iklim historis sampai 2025 yang memiliki fitur kosong: {len(climate_hist_missing):,}.",
        f"- Demand region-tahun dengan tepat satu nilai bulanan unik: "
        f"{int((demand_variability['unique_monthly_demand'] == 1).sum()):,} dari "
        f"{len(demand_variability):,} kelompok region-tahun.",
        "",
        "## Implikasi metodologis",
        "- Supply dapat diuji sebagai panel bulanan per wilayah setelah penanganan nilai nol diverifikasi.",
        "- Demand saat ini dibentuk dari kebutuhan tahunan yang dibagi rata menjadi 12 bulan. "
        "Karena itu, seri bulanan bukan observasi konsumsi aktual bulanan dan memiliki nilai konstan "
        "dalam setiap region-tahun. Forecast demand 1/3/6 bulan dari seri ini harus diberi label "
        "estimasi/proyeksi berbasis asumsi, bukan realisasi konsumsi bulanan.",
        "- Data iklim historis 2020-2025 dapat dipakai untuk eksperimen fitur iklim. Untuk prediksi "
        "multi-step, gunakan hanya iklim masa lalu/lag yang tersedia pada waktu prediksi, kecuali "
        "tersedia prakiraan iklim masa depan. Jangan memasukkan realisasi iklim bulan target.",
        "- Validasi harus berbasis waktu. Jangan melakukan random train-test split pada data deret waktu.",
        "",
        "## Batasan",
        "- Audit ini memeriksa struktur, kelengkapan, duplikasi, rentang waktu, nilai nol/negatif, "
        "dan variasi demand; audit ini tidak membuktikan validitas substantif sumber data.",
        "- Tidak ada imputasi atau pengubahan nilai dilakukan oleh script ini.",
    ]
    (OUT / "audit_report.md").write_text("\n".join(report), encoding="utf-8")

    print("AUDIT SELESAI")
    print(f"Supply: {len(supply):,} rows | {supply['region_standard'].nunique()} regions | "
          f"{supply['period'].min()} - {supply['period'].max()} | duplicates={supply_dups}")
    print(f"Demand: {len(demand):,} rows | {demand['region_standard'].nunique()} regions | "
          f"{demand['period'].min()} - {demand['period'].max()} | duplicates={demand_dups}")
    print(f"Climate: {len(climate):,} rows | {climate['region'].nunique()} regions | "
          f"{climate['period'].min()} - {climate['period'].max()} | duplicates={climate_dups}")
    print(f"Zero production records: {len(zero_supply)}")
    print(f"Climate incomplete rows (all periods): {len(climate_missing)}")
    print(f"Climate incomplete rows (through 2025): {len(climate_hist_missing)}")
    print(f"Output: {OUT}")

if __name__ == "__main__":
    main()