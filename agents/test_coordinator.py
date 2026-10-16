
from pathlib import Path

from data_processing.loader import (
    load_produksi,
    load_distribusi
)

from data_processing.preprocessing import prepare_production
from agents.supply_agent import SupplyAgent
from agents.distribution_agent import DistributionAgent
from agents.coordinator_agent import CoordinatorAgent


def main():
    production = prepare_production(load_produksi())
    supply = SupplyAgent().run(production)

    distribution_raw = load_distribusi()
    distribution = DistributionAgent().run(distribution_raw)

    coordinator = CoordinatorAgent()

    result = coordinator.run({
        "supply": supply,
        "demand": None,
        "distribution": distribution
    })

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    result["supply_demand"].to_csv(
        output_dir / "coordinator_supply_demand.csv",
        index=False
    )

    if result["distribution_summary"] is not None:
        result["distribution_summary"].to_csv(
            output_dir / "coordinator_distribution_summary.csv",
            index=False
        )

    print("\n=== COORDINATOR AGENT ===")
    print("Baris supply:", len(supply))
    print("Baris hasil integrasi:", len(result["supply_demand"]))
    print("\nHasil integrasi:")
    print(result["supply_demand"].head().to_string(index=False))

    print("\nRingkasan distribusi:")
    print(result["distribution_summary"])

    print("\nOutput tersimpan di:", output_dir)


if __name__ == "__main__":
    main()
