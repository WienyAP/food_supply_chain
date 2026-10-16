
from pathlib import Path

from data_processing.loader import (
    load_produksi,
    load_distribusi
)

from data_processing.preprocessing import prepare_production
from agents.supply_agent import SupplyAgent
from agents.distribution_agent import DistributionAgent
from agents.recommendation_agent import RecommendationAgent


def main():
    production = prepare_production(load_produksi())
    supply = SupplyAgent().run(production)

    distribution = DistributionAgent().run(
        load_distribusi()
    )

    recommendation = RecommendationAgent().run({
        "supply": supply,
        "distribution": distribution
    })

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    recommendation.to_csv(
        output_dir / "recommendation_candidates.csv",
        index=False
    )

    print("\n=== RECOMMENDATION AGENT ===")
    print("Jumlah kandidat:", len(recommendation))

    if not recommendation.empty:
        print(recommendation.head(10).to_string(index=False))
    else:
        print("Tidak ditemukan kandidat pada data yang tersedia.")

    print("\nOutput tersimpan di:", output_dir)


if __name__ == "__main__":
    main()
