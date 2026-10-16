
from pathlib import Path
import pandas as pd

from data_processing.loader import (
    load_produksi,
    load_konsumsi,
    load_penduduk,
    load_distribusi
)

from data_processing.preprocessing import (
    prepare_production,
    prepare_consumption,
    prepare_population
)

from agents.agent_graph import build_agri_food_graph

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Satuan konsumsi belum terverifikasi
CONSUMPTION_UNIT = None

def main():
    print("=" * 60)
    print("AGRI-FOOD SUPPLY CHAIN - MULTI-AGENT PROTOTYPE")
    print("LangGraph + Ollama")
    print("=" * 60)

    # 1. Load dan preprocessing produksi
    print("\n[1/6] Memuat dan memvalidasi data produski...")
    production = prepare_production(load_produksi())

    # 2. Load dan preprocessing konsumsi
    print("\n[2/6] Memuat data konsumsi dan penduduk...")
    consumption = prepare_consumption(load_konsumsi())
    population = prepare_population(load_penduduk())

    # 3. Load data distribusi
    print("\n[3/6] Memuat data distribusi...")
    distribution_raw = load_distribusi()

    print("Baris produksi bulanan: ", len(production))
    print("Baris konsumsi beras: ", len(consumption))
    print("Baris penduduk: ", len(population))
    print("Baris distribusi mentah: ", len(distribution_raw))

    # 4. Jalankan workflow LangGraph
    print("\n[4/6] Menjalankan workflow LangGraph...")
    graph = build_agri_food_graph()

    initial_state = {
        "production": production,
        "consumption": consumption,
        "population": population,
        "distribution_raw": distribution_raw,
        "consumption_unit": "kg_per_capita_per_year",
    }

    result = graph.invoke(initial_state)

    supply_result = result["supply_result"]
    demand_result = result.get("demand_result")

    print("\n--- CEK KUNCI WILAYAH ---")
    print("Supply:")
    print(supply_result[["year", "region", "region_key"]].head(5).to_string(index=False))

    print("\nDemand:")
    print(demand_result[["year", "region", "region_key"]].head(5).to_string(index=False))

    print("\nJumlah region_key supply:", supply_result["region_key"].nunique())
    print("Jumlah region_key demand:", demand_result["region_key"].nunique())
    print(
        "Kunci wilayah yang cocok:",
        len(set(supply_result["region_key"]) & set(demand_result["region_key"]))
    )

    distribution_result = result["distribution_result"]
    coordinator_result = result["coordinator_result"]
    recommendation_result = result["recommendation_result"]

    supply_demand = coordinator_result["supply_demand"]
    distribution_summary = coordinator_result.get("distribution_summary")

    # 5. Simpan output
    print("[5/6] Menyimpan hasil analisis...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pertahankan lokasi dan nama file hasil pipeline sebelumnya
    supply_demand.to_csv(
        DATA_DIR / "coordinator_supply_demand.csv",
        index=False,
    )

    if distribution_summary is not None:
        distribution_summary.to_csv(
            DATA_DIR / "coordinator_distribution_summary.csv",
            index=False,
        )

    # Output tambahan tiap agen.
    supply_result.to_csv(
        OUTPUT_DIR / "supply_result.csv",
        index=False,
    )

    distribution_result.to_csv(
        OUTPUT_DIR / "distribution_processed.csv",
        index=False,
    )

    recommendation_result.to_csv(
        OUTPUT_DIR / "recommendation_candidates.csv",
        index=False
    )

    if demand_result is not None:
        demand_result.to_csv(
            OUTPUT_DIR / "demand_result.csv",
            index=False
        )

    # Simpan laporan LLM, jika node OLlama tersedia.
    llm_report = result.get("llm_report")

    if llm_report:
        (OUTPUT_DIR / "llm_report.md").write_text(
            llm_report,
            encoding="utf-8",
        )

    # 6. Ringkasan hasil
    print("\n[6/6] Memeriksa hasil workflow...")
    print("\n" + "=" * 60)
    print("PIPELINE SELESAI")
    print("="  * 60)

    print("Baris supply:", len(supply_result))
    print("Wilayah supply:", supply_result["region_key"].nunique())
    print("Tahun produksi:", sorted(supply_result["year"].unique()))
    print("Catatan distribusi beras:", len(distribution_result))
    print("Kandidat rekomendasi:", len(recommendation_result))

    if demand_result is None:
        print(
            "Estimasi konsumsi: belum dihitung karena satuan"
            "Konsumsi_Pangan belum terverifikasi."
        )
    else:
        print("Baris estimasi konsumsi: ", len(demand_result))

    print("\nFile hasil utama:")
    print(DATA_DIR / "coordinator_supply_demand.csv")
    print(DATA_DIR / "coordinator_distribution_summary.csv")

    print("\nFile hasil tambahan:")
    print(OUTPUT_DIR / "supply_result.csv")
    print(OUTPUT_DIR / "distribution_processed.csv")
    print(OUTPUT_DIR / "recommendation_candidates.csv")

    if llm_report:
        print(OUTPUT_DIR / "llm_report.md")
    else:
        print("Laporan Ollama belum tersedia")


    print("\nContoh hasil integrasi:")
    display_columns = [
        col for col in [
            "year",
            "region",
            "produksi_beras",
            "estimasi_konsumsi_ton_tahun",
            "supply_status",
        ]
        if col in supply_demand.columns
    ]

    if display_columns:
        print(
            supply_demand[display_columns]
            .head(10)
            .to_string(index=False)
        )

    print("\nCatatan: hasil produksi-konsumsi bukan NBM lengkap")


if __name__ == "__main__":
    main()