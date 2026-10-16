
import pandas as pd
from agents.base_agent import BaseAgent


class DemandAgent(BaseAgent):

    def __init__(self):
        super().__init__("DemandAgent")

    def run(
        self,
        data: dict,
    ) -> pd.DataFrame:

        consumption = data["consumption"].copy()
        population = data["population"].copy()

        required_consumption = {
            "year",
            "region",
            "region_key",
            "Konsumsi_Pangan"
        }

        required_population = {
            "year",
            "region_key",
            "jumlah_penduduk"
        }

        missing_c = required_consumption - set(
            consumption.columns
        )

        missing_p = required_population - set(
            population.columns
        )

        if missing_c:
            raise ValueError(
                f"Kolom konsumsi tidak tersedia: {sorted(missing_c)}"
            )

        if missing_p:
            raise ValueError(
                f"Kolom penduduk tidak tersedia: {sorted(missing_p)}"
            )

        result = consumption.merge(
            population[
                ["year", "region_key", "jumlah_penduduk"]
            ],
            on=["year", "region_key"],
            how="left",
            validate="one_to_one"
        )

        if result["jumlah_penduduk"].isna().any():
            missing_regions = result.loc[
                result["jumlah_penduduk"].isna(),
                ["year", "region"]
            ]

            raise ValueError(
                "Data penduduk tidak cocok dengan data konsumsi:\n"
                + missing_regions.to_string(index=False)
            )

        # Pastikan satuan konsumsi telah diverifikasi.
        consumption_unit = data.get("consumption_unit")

        if consumption_unit != "kg_per_capita_per_year":
            raise ValueError(
                "Satuan Konsumsi_Pangan belum dikonfirmasi sebagai "
                "kg/kapita/tahun. Verifikasi metadata sumber dahulu."
            )

        result["konsumsi_per_kapita_kg_tahun"] = (
            result["Konsumsi_Pangan"]
        )

        result["estimasi_konsumsi_ton_tahun"] = (
            result["konsumsi_per_kapita_kg_tahun"]
            * result["jumlah_penduduk"]
            / 1000
        )

        result["unit_status"] = "verified_from_source_metadata"

        return result[
            [
                "year",
                "region",
                "region_key",
                "jumlah_penduduk",
                "konsumsi_per_kapita_kg_tahun",
                "estimasi_konsumsi_ton_tahun",
                "unit_status"
            ]
        ].reset_index(drop=True)
