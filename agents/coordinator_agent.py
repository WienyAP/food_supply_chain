
import pandas as pd
from .base_agent import BaseAgent


class CoordinatorAgent(BaseAgent):
    """Mengintegrasikan hasil supply, demand, dan distribusi."""

    def __init__(self):
        super().__init__("CoordinatorAgent")

    def run(self, data: dict) -> dict:
        supply = data["supply"].copy()
        demand = data.get("demand")
        distribution = data.get("distribution")

        # Validasi kolom kunci supply.
        supply_keys = {"year", "region_key"}
        missing_supply = supply_keys - set(supply.columns)

        if missing_supply:
            raise ValueError(
                f"Kolom kunci supply tidak tersedia: "
                f"{sorted(missing_supply)}"
            )

        if supply.duplicated(["year", "region_key"]).any():
            raise ValueError(
                "Supply memiliki duplikasi tahun-wilayah."
            )

        # Integrasi supply dan demand hanya jika demand tersedia.
        if demand is not None:
            demand = demand.copy()

            demand_keys = {"year", "region_key"}
            missing_demand = demand_keys - set(demand.columns)

            if missing_demand:
                raise ValueError(
                    f"Kolom kunci demand tidak tersedia: "
                    f"{sorted(missing_demand)}"
                )

            if demand.duplicated(["year", "region_key"]).any():
                raise ValueError(
                    "Demand memiliki duplikasi tahun-wilayah."
                )

            merged = supply.merge(
                demand,
                on=["year", "region_key"],
                how="left",
                suffixes=("_supply", "_demand"),
                validate="one_to_one"
            )
        else:
            merged = supply.copy()

        # Data distribusi disimpan sebagai ringkasan terpisah.
        # Distribusi belum tentu lengkap untuk semua wilayah dan tahun.
        distribution_summary = None

        if distribution is not None:
            distribution = distribution.copy()

            required_distribution = {
                "Tahun",
                "volume_kg",
                "volume_ton"
            }

            missing_distribution = (
                required_distribution - set(distribution.columns)
            )

            if missing_distribution:
                raise ValueError(
                    "Kolom distribusi tidak tersedia: "
                    f"{sorted(missing_distribution)}"
                )

            distribution_summary = (
                distribution.groupby(
                    "Tahun",
                    as_index=False
                )
                .agg(
                    jumlah_catatan=("volume_kg", "size"),
                    total_volume_kg=("volume_kg", "sum"),
                    total_volume_ton=("volume_ton", "sum")
                )
                .rename(columns={"Tahun": "year"})
            )

        return {
            "supply_demand": merged,
            "distribution_summary": distribution_summary
        }
