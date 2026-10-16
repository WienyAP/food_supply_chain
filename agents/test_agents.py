
from data_processing.loader import (
    load_produksi,
    load_konsumsi,
    load_penduduk
)

from data_processing.preprocessing import (
    prepare_production,
    prepare_consumption,
    prepare_population
)

from agents.supply_agent import SupplyAgent
from agents.demand_agent import DemandAgent


def main():
    production = prepare_production(load_produksi())
    consumption = prepare_consumption(load_konsumsi())

    print("\n=== PEMERIKSAAN DATA KONSUMSI ===")
    print(
        consumption[
            ["year", "region", "comodity", "Konsumsi_Pangan"]
        ].head(10).to_string(index=False)
    )

    print("\nStatistik nilai konsumsi:")
    print(consumption["Konsumsi_Pangan"].describe())

    population = prepare_population(load_penduduk())

    supply_result = SupplyAgent().run(production)

    print("\n=== SUPPLY AGENT ===")
    print(supply_result.head())
    print("Jumlah baris:", len(supply_result))

    # Isi nilai berikut hanya setelah satuan konsumsi diverifikasi.
    consumption_unit = None

    if consumption_unit is None:
        print(
            "\nDemand Agent belum dijalankan: "
            "satuan Konsumsi_Pangan belum dikonfirmasi."
        )
        return

    demand_result = DemandAgent().run({
        "consumption": consumption,
        "population": population,
        "consumption_unit": consumption_unit
    })

    print("\n=== DEMAND AGENT ===")
    print(demand_result.head())
    print("Jumlah baris:", len(demand_result))


if __name__ == "__main__":
    main()
