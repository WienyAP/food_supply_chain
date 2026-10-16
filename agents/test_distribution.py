
from data_processing.loader import load_distribusi
from agents.distribution_agent import DistributionAgent
from pathlib import Path

def main():
    raw = load_distribusi()

    agent = DistributionAgent()
    result = agent.run(raw)

    route_summary = agent.summarize_by_route(result)
    monthly_summary = agent.summarize_monthly(result)
    location_summary = agent.summarize_by_location(result)

    print("\n=== DISTRIBUTION AGENT ===")
    print("Jumlah catatan beras:", len(result))
    print("Total volume (ton):", result["volume_ton"].sum())

    print("\n=== RINGKASAN RUTE PROVINSI ===")
    print(route_summary.head(10).to_string(index=False))

    print("\n=== RINGKASAN BULANAN ===")
    print(monthly_summary.head(10).to_string(index=False))

    print("\n=== RINGKASAN LOKASI ===")
    print(location_summary.head(10).to_string(index=False))

    
    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(
        output_dir / "distribution_processed.csv",
        index=False
    )

    route_summary.to_csv(
        output_dir / "distribution_route_summary.csv",
        index=False
    )

    monthly_summary.to_csv(
        output_dir / "distribution_monthly_summary.csv",
        index=False
    )

    location_summary.to_csv(
        output_dir / "distribution_location_summary.csv",
        index=False
    )

    print("\nOutput tersimpan di:", output_dir)


if __name__ == "__main__":
    main()
